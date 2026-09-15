#!/usr/bin/env python3
"""Fig. 5-style NIQE histograms for EM volumes.

This is intentionally a curve-only companion to evaluate_zstripe_benchmark.py.
It follows the EMISAC Fig. 5 sampling logic: score every available XZ and YZ
cross-section of a held-out volume, then plot the score histograms.  XY is
kept to one reproducibly random held-out slice when requested by the user.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np

from .niqe import fit_pristine_model, niqe_score, niqe_scores_gpu, summary, torch
from .benchmark import (
    COLORS, LABELS, MATCHED_METHOD_BY_SOURCE, METHODS, PLOT_LABELS,
    add_original_global_matches, cross_section_at, ground_truth, load,
    stacks_for_dataset, uses_z6_resampling,
)


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def cached_axis_scores(volume: np.ndarray, axis: str, mu: np.ndarray, cov: np.ndarray,
                       patch: int, z_ratio: float, border: int, workers: int,
                       cache_path: Path, stride: int = 1, engine: str = "cpu",
                       gpu_batch_size: int = 8, max_positions: int = 0) -> Tuple[List[int], List[float]]:
    """Score every interior XZ or YZ plane, checkpointing ordered results.

    ``ThreadPoolExecutor`` is useful here because OpenCV/SciPy/NumPy release
    the GIL for the convolution, special-function and linear-algebra work.
    It avoids copying a 1400-slice volume into multiple Windows processes.
    """
    side = volume.shape[1] if axis == "xz" else volume.shape[2]
    positions = list(range(border, side - border, stride))
    if max_positions > 0 and len(positions) > max_positions:
        # Uniform interior sampling retains the full lateral extent while
        # bounding the cost of paper-style NIQE distributions.
        positions = [positions[i] for i in np.linspace(0, len(positions) - 1, max_positions).round().astype(int)]
    scores: List[float] = []
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            cached_positions = data["positions"].astype(int).tolist()
            if cached_positions == positions:
                scores = data["scores"].astype(float).tolist()
        except (OSError, KeyError, ValueError):
            scores = []
    scores = scores[:len(positions)]

    def score_at(position: int) -> float:
        section = cross_section_at(volume, axis, position, z_ratio, border)
        return float(niqe_score(section, mu, cov, patch))

    remaining = positions[len(scores):]
    if engine == "gpu":
        for start in range(0, len(remaining), gpu_batch_size):
            sections = [cross_section_at(volume, axis, position, z_ratio, border) for position in remaining[start:start + gpu_batch_size]]
            scores.extend(niqe_scores_gpu(sections, mu, cov, patch, batch_size=gpu_batch_size))
            np.savez_compressed(cache_path, positions=np.asarray(positions), scores=np.asarray(scores, dtype=np.float64))
        return positions, scores
    if workers <= 1:
        iterator: Iterable[float] = map(score_at, remaining)
        executor = None
    else:
        executor = ThreadPoolExecutor(max_workers=workers, thread_name_prefix="niqe")
        iterator = executor.map(score_at, remaining)
    try:
        for number, score in enumerate(iterator, start=len(scores) + 1):
            scores.append(float(score))
            if number % 16 == 0:
                np.savez_compressed(cache_path, positions=np.asarray(positions), scores=np.asarray(scores, dtype=np.float64))
    finally:
        if executor is not None:
            executor.shutdown(wait=True)
    np.savez_compressed(cache_path, positions=np.asarray(positions), scores=np.asarray(scores, dtype=np.float64))
    return positions, scores


def random_xy_score(stack, test_start: int, seed: int, mu: np.ndarray, cov: np.ndarray, patch: int,
                    cache_path: Path, engine: str = "cpu", gpu_batch_size: int = 8) -> Tuple[int, float]:
    rng = np.random.default_rng(seed)
    z_index = int(rng.integers(test_start, stack.count))
    if cache_path.exists():
        try:
            data = np.load(cache_path)
            if int(data["z_index"]) == z_index:
                return z_index, float(data["score"])
        except (OSError, KeyError, ValueError):
            pass
    image = stack.read(z_index)
    score = float(niqe_scores_gpu([image], mu, cov, patch, batch_size=gpu_batch_size)[0]) if engine == "gpu" else float(niqe_score(image, mu, cov, patch))
    np.savez_compressed(cache_path, z_index=np.asarray(z_index), score=np.asarray(score))
    return z_index, score


def histogram_page(dataset: str, profiles: Dict[str, Dict[str, object]], methods: Sequence[str],
                   output: Path, pdf: PdfPages, suffix: str) -> List[Dict[str, object]]:
    plt.rcParams.update({"font.family": "serif", "font.serif": ["Times New Roman", "DejaVu Serif"],
                         "font.size": 9, "axes.linewidth": .8, "figure.dpi": 150, "savefig.dpi": 300})
    fig, axes = plt.subplots(1, 3, figsize=(11.1, 3.25))
    fig.subplots_adjust(left=.065, right=.995, bottom=.19, top=.79, wspace=.18)
    samples: List[Dict[str, object]] = []
    planes = (("xz", "X-Z cross-sections quality"), ("yz", "Y-Z cross-sections quality"), ("xy", "X-Y slices quality"))
    for ax, (plane, title) in zip(axes, planes):
        values = {method: np.asarray(profiles[method][plane], dtype=float) for method in methods}
        valid = [v[np.isfinite(v)] for v in values.values() if np.any(np.isfinite(v))]
        all_values = np.concatenate(valid) if valid else np.empty(0)
        if not len(all_values):
            ax.set_title(title); continue
        lower, upper = np.percentile(all_values, [0, 99.5])
        pad = max((upper - lower) * .06, .05)
        bins = np.linspace(max(0, lower - pad), upper + pad, 41)
        for method, value in values.items():
            value = value[np.isfinite(value)]
            if not len(value):
                continue
            counts, edges = np.histogram(value, bins=bins)
            centers = (edges[:-1] + edges[1:]) / 2
            ax.plot(centers, counts, drawstyle="steps-mid", lw=1.25, color=COLORS[method], label=PLOT_LABELS[method])
            for score in value:
                samples.append({"dataset": dataset, "group": suffix, "method": method, "plane": plane,
                                "niqe_score": float(score), "sample_count_for_method_plane": len(value)})
        ax.set_title(title, fontweight="bold", fontsize=10)
        ax.set_xlabel("NIQE score"); ax.set_ylabel("Count")
        ax.grid(alpha=.18, linewidth=.45); ax.legend(loc="upper right", fontsize=6.3, frameon=True, fancybox=False)
        if plane == "xy":
            ax.text(.02, .96, "One random held-out XY slice / method", transform=ax.transAxes,
                    va="top", fontsize=6.5, color="#444")
    fig.suptitle(f"{dataset}: histogram of held-out NIQE scores ({suffix})", fontweight="bold", y=.965)
    stem = output / f"niqe_figure5_histogram_{dataset}_{suffix.replace(' ', '_')}"
    fig.savefig(stem.with_suffix(".png"), dpi=350); fig.savefig(stem.with_suffix(".pdf")); pdf.savefig(fig); plt.close(fig)
    return samples


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Fast Fig. 5-style EM NIQE histogram only")
    parser.add_argument("--datasets", default="feijunyun")
    parser.add_argument("--methods", default="", help="Comma-separated raw methods; empty evaluates all available methods.")
    parser.add_argument("--output-dir", type=Path, default=Path("niqe_figure5_histograms"))
    parser.add_argument("--train-first-slices", type=int, default=200)
    parser.add_argument("--test-z-start", type=int, default=200)
    parser.add_argument("--xy-seed", type=int, default=20260721)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--sharpness-fraction", type=float, default=.75)
    parser.add_argument("--cross-plane-border", type=int, default=32)
    parser.add_argument("--cross-section-stride", type=int, default=1, help="1 scores every interior cross-section (paper setting).")
    parser.add_argument("--workers", type=int, default=0, help="0 selects a conservative CPU-thread count.")
    parser.add_argument("--engine", choices=("auto", "cpu", "gpu"), default="auto", help="auto uses CUDA when available.")
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--with-global-match", action="store_true", help="Also plot five Original-global-matched method outputs.")
    args = parser.parse_args()
    if args.train_first_slices < 1 or args.test_z_start < args.train_first_slices:
        raise ValueError("The held-out test start must be at or after the NIQE training-slice count.")
    if args.cross_section_stride < 1:
        raise ValueError("--cross-section-stride must be at least 1.")
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True); cache_dir = output / "niqe_cache"; cache_dir.mkdir(exist_ok=True)
    workers = args.workers if args.workers > 0 else min(8, max(1, (os.cpu_count() or 4) // 2))
    cuda_available = bool(torch is not None and torch.cuda.is_available())
    engine = "gpu" if args.engine == "auto" and cuda_available else args.engine
    if engine == "gpu" and not cuda_available:
        raise RuntimeError("--engine gpu requires CUDA PyTorch. Activate stripe_removal or select --engine cpu.")
    all_samples: List[Dict[str, object]] = []; all_summary: List[Dict[str, object]] = []; metadata: Dict[str, object] = {"arguments": vars(args), "workers": workers, "engine": engine, "cuda_available": cuda_available, "datasets": {}}
    requested = [x for x in args.datasets.split(",") if x]
    requested_methods = {x for x in args.methods.split(",") if x}
    with PdfPages(output / "niqe_figure5_histograms.pdf") as pdf:
        for dataset in requested:
            stacks = stacks_for_dataset(root, dataset)
            if stacks.get("Original") is None:
                raise ValueError(f"{dataset}: Original stack is unavailable.")
            gt = ground_truth(root, dataset)
            if gt is None:
                raise ValueError(f"{dataset}: a registered clean TIFF is required for per-dataset NIQE.")
            if args.test_z_start >= stacks["Original"].count:
                raise ValueError(f"{dataset}: no held-out test slices remain.")
            train_ids = list(range(min(gt.count, args.train_first_slices)))
            model_path = cache_dir / f"{dataset}_pristine_model_{len(train_ids)}slices_patch{args.patch_size}.npz"
            if model_path.exists():
                saved = np.load(model_path); mu, cov, patches = saved["mu"], saved["cov"], int(saved["patches"])
            else:
                mu, cov, patches = fit_pristine_model((gt.read(i) for i in train_ids), args.patch_size, args.sharpness_fraction)
                np.savez_compressed(model_path, mu=mu, cov=cov, patches=np.asarray(patches))
            match_sources = tuple(m for m in MATCHED_METHOD_BY_SOURCE if not requested_methods or m in requested_methods) if args.with_global_match else ()
            matches = add_original_global_matches(stacks, match_sources, args.test_z_start)
            methods = [method for method in METHODS if stacks.get(method) is not None and (not requested_methods or method in requested_methods)]
            matched_methods = ["Original"] + [entry["method"] for entry in matches]
            ratio = 6.0 if uses_z6_resampling(dataset) else 1.0
            gpu_validation = None
            if engine == "gpu":
                validation_ids = [args.test_z_start, min(stacks["Original"].count - 1, args.test_z_start + 700)]
                validation_images = [stacks["Original"].read(index) for index in validation_ids]
                cpu_values = [niqe_score(image, mu, cov, args.patch_size) for image in validation_images]
                gpu_values = niqe_scores_gpu(validation_images, mu, cov, args.patch_size, batch_size=args.gpu_batch_size)
                gpu_validation = {"z_indices": validation_ids, "cpu": cpu_values, "gpu": gpu_values,
                                  "max_absolute_difference": max(abs(a - b) for a, b in zip(cpu_values, gpu_values))}
            profiles: Dict[str, Dict[str, object]] = {}
            for method in methods + [m for m in matched_methods if m != "Original"]:
                stack = stacks[method]
                tag = f"{dataset}_{method}_z{args.test_z_start}_border{args.cross_plane_border}_patch{args.patch_size}_{engine}"
                xy_z, xy_score = random_xy_score(stack, args.test_z_start, args.xy_seed, mu, cov, args.patch_size, cache_dir / f"{tag}_xy_seed{args.xy_seed}.npz", engine, args.gpu_batch_size)
                volume = load(stack, list(range(args.test_z_start, stack.count)))
                xz_pos, xz_scores = cached_axis_scores(volume, "xz", mu, cov, args.patch_size, ratio, args.cross_plane_border, workers, cache_dir / f"{tag}_xz_stride{args.cross_section_stride}.npz", args.cross_section_stride, engine, args.gpu_batch_size)
                yz_pos, yz_scores = cached_axis_scores(volume, "yz", mu, cov, args.patch_size, ratio, args.cross_plane_border, workers, cache_dir / f"{tag}_yz_stride{args.cross_section_stride}.npz", args.cross_section_stride, engine, args.gpu_batch_size)
                del volume
                profiles[method] = {"xy": [xy_score], "xz": xz_scores, "yz": yz_scores, "xy_z_index": xy_z}
                for plane, positions, values in (("xy", [xy_z], [xy_score]), ("xz", xz_pos, xz_scores), ("yz", yz_pos, yz_scores)):
                    record = {"dataset": dataset, "method": method, "plane": plane, **summary(values),
                              "xy_random_z_index": xy_z, "test_z_start": args.test_z_start,
                              "test_z_count": stack.count - args.test_z_start, "cross_plane_border": args.cross_plane_border,
                              "z_resampling_for_cross_sections": ratio, "position_count": len(positions), "engine": engine}
                    all_summary.append(record)
            all_samples += histogram_page(dataset, profiles, methods, output, pdf, "raw_methods")
            if len(matched_methods) > 1:
                all_samples += histogram_page(dataset, profiles, matched_methods, output, pdf, "original_global_matched")
            metadata["datasets"][dataset] = {"clean_tiff": str(gt.path), "training_slice_ids": train_ids,
                                               "selected_pristine_patches": patches, "xy_random_z_index": profiles["Original"]["xy_z_index"],
                                               "test_z_count": stacks["Original"].count - args.test_z_start,
                                               "z_ratio": ratio, "interior_cross_sections_per_axis": len(profiles["Original"]["xz"]),
                                               "gpu_validation": gpu_validation}
    write_csv(output / "niqe_figure5_samples.csv", all_samples)
    write_csv(output / "niqe_figure5_summary.csv", all_summary)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    print(f"Finished Fig. 5-style NIQE curves. PDF: {output / 'niqe_figure5_histograms.pdf'}")


if __name__ == "__main__":
    main()
