#!/usr/bin/env python3
"""Read-only saved-output benchmark: no photometric fitting or normalization.

Real volumes use an automatically audited *contiguous-prefix* normal-image
pool for the NIQE pristine model.  The held-out tail remains a completely
continuous Z sequence: selected model images are copied, never removed from
their source directory.  Selection is constrained to the densest compact
brightness band, so an NIQE prior is never formed from visibly mismatched
whole-slice luminance levels.  Simulated volumes use their registered clean
TIFFs.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import subprocess
import sys
import json
import math
import shutil
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import cv2
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
import numpy as np
from scipy.ndimage import gaussian_filter1d

from .niqe import fit_pristine_model, niqe_score, niqe_scores_gpu, torch
from .niqe_cross_sections import cached_axis_scores
from .benchmark import (
    COLORS, METHODS, PLOT_LABELS, PNGStack, VolumeTIFF, PagesTIFF,
    comparison_page, curves_page, full_reference, ground_truth, load, style,
    make_brisque, nonreference, specialized_table_page, spaced, style, table_page,
    uses_z6_resampling, stacks_for_dataset, configured_root,
)

KIDNEY_SIMULATED = {
    "feijunyun",
    "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600-weak",
    "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600-medium",
    "jrc_mus-kidney_em-fibsem-uint8-s0_uint8_x10000-11024_y5000-6024_z5000-6600-strong",
}
SIMULATED = KIDNEY_SIMULATED | {"SIM-jrc_mus-liver-3"}
# A NIQE prior must be trained on the same acquisition domain.  In particular,
# SNEMItrain now uses only its own raw slices (not SNEMItest).
SNEMI_DONOR: Dict[str, str] = {}
PRISTINE_SELECTION_SCHEMA = 2
METHOD_CACHE_SCHEMA = 3


class WindowStack:
    """Read-only continuous Z window over a source stack.

    The wrapper deliberately reindexes the selected source interval to
    ``0..count-1``.  Therefore every cross-section, profile, full-reference
    score is calculated from exactly the same uninterrupted
    held-out interval, without ever exposing NIQE-prior frames to testing.
    """
    def __init__(self, source, start: int, stop: int):
        if not (0 <= start < stop <= source.count):
            raise ValueError(f"Invalid stack window [{start}, {stop}) for {source.count} slices")
        self.source, self.start, self.stop = source, int(start), int(stop)
        self.count = self.stop - self.start

    def read(self, index: int) -> np.ndarray:
        if not 0 <= index < self.count:
            raise IndexError(index)
        return self.source.read(self.start + int(index))

    def read_many(self, indices: Sequence[int]) -> np.ndarray:
        ids = [int(i) for i in indices]
        if any(i < 0 or i >= self.count for i in ids):
            raise IndexError("WindowStack index outside held-out Z interval")
        return self.source.read_many([self.start + i for i in ids])


def stable_hash(payload: object) -> str:
    """Stable content key used to invalidate only the affected cached method."""
    text = json.dumps(payload, sort_keys=True, ensure_ascii=False, default=str, separators=(",", ":"))
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def stack_fingerprint(stack) -> Dict[str, object]:
    """Cheap, provenance-preserving description of a stack and its source files."""
    if isinstance(stack, WindowStack):
        return {"kind": "window", "start": stack.start, "stop": stack.stop,
                "source": stack_fingerprint(stack.source)}
    if isinstance(stack, PNGStack):
        files = [{"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                 for path in stack.paths]
        return {"kind": "png", "count": stack.count, "files": files}
    if isinstance(stack, VolumeTIFF):
        path = stack.path; stat = path.stat()
        return {"kind": "volume_tiff", "count": stack.count, "path": str(path),
                "size": stat.st_size, "mtime_ns": stat.st_mtime_ns}
    if isinstance(stack, PagesTIFF):
        files = [{"path": str(path), "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
                 for path in stack.paths]
        return {"kind": "pages_tiff", "count": stack.count, "files": files}
    return {"kind": type(stack).__name__, "count": getattr(stack, "count", None)}


def prepare_saved_outputs(stacks, methods, start, stop, ground_truth_stack=None):
    """Expose the same saved pixels in a shared Z window; never fit intensities.

    vEMDestripe must already have completed its reference-calibration/export
    stage. Other methods are their delivered outputs. Ground truth, if supplied,
    is only a scoring target, never a source of evaluator-side parameters.
    """
    selected = {name: WindowStack(stacks[name], start, stop)
                for name in methods if stacks.get(name) is not None and stacks[name].count >= stop}
    target = WindowStack(ground_truth_stack, start, stop) if ground_truth_stack is not None else None
    return selected, target


def method_cache_key(dataset: str, method: str, stack, niqe_model_path: Path, args, z_ratio: float,
                     score_protocol: Dict[str, object]) -> str:
    protocol = {
        "schema": METHOD_CACHE_SCHEMA, "dataset": dataset, "method": method,
        "stack": stack_fingerprint(stack),
        "model": {"path": str(niqe_model_path), "size": niqe_model_path.stat().st_size,
                  "mtime_ns": niqe_model_path.stat().st_mtime_ns},
        "score_protocol": score_protocol,
        "settings": {"iqa_slices": args.iqa_slices, "metric_cross_sections": args.metric_cross_sections,
                     "niqe_cross_sections": args.niqe_cross_sections, "niqe_xy_slices": args.niqe_xy_slices,
                     "cross_plane_border": args.cross_plane_border, "gpu_batch_size": args.gpu_batch_size,
                     "z_ratio": z_ratio},
    }
    return stable_hash(protocol)


def load_method_cache(cache_root: Path, key: str) -> Optional[Tuple[Dict[str, object], Dict[str, object]]]:
    folder = cache_root / key
    metrics_path, profile_path = folder / "metrics.json", folder / "profile.npz"
    if not (metrics_path.is_file() and profile_path.is_file()):
        return None
    try:
        metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
        arrays = np.load(profile_path)
        profile = {name: arrays[name].astype(float).tolist() for name in arrays.files}
        return metrics, profile
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return None


def save_method_cache(cache_root: Path, key: str, metrics: Dict[str, object], profile: Dict[str, object],
                      provenance: Dict[str, object]) -> None:
    folder = cache_root / key; folder.mkdir(parents=True, exist_ok=True)
    scalar_metrics = {k: (float(v) if isinstance(v, (np.floating, np.integer)) else v) for k, v in metrics.items()}
    (folder / "metrics.json").write_text(json.dumps(scalar_metrics, ensure_ascii=False, indent=2, default=str), encoding="utf-8")
    arrays = {key: np.asarray(value, dtype=float) for key, value in profile.items()}
    np.savez_compressed(folder / "profile.npz", **arrays)
    (folder / "provenance.json").write_text(json.dumps(provenance, ensure_ascii=False, indent=2, default=str), encoding="utf-8")


def paper_niqe_curve_page(dataset: str, profiles: Dict[str, Dict[str, object]], methods: Sequence[str],
                          figures: Path, pdf: PdfPages, group: str) -> List[Dict[str, object]]:
    """Fig.-5-like smoothed histogram curves plus persisted bin-level data."""
    style(); fig, axes = plt.subplots(1, 3, figsize=(11.3, 3.15), constrained_layout=True)
    rows: List[Dict[str, object]] = []
    planes = (("niqe_xz_samples", "X-Z cross-sections quality", "xz"),
              ("niqe_yz_samples", "Y-Z cross-sections quality", "yz"),
              ("niqe_xy_samples", "X-Y slices quality", "xy"))
    for ax, (key, label, code) in zip(axes, planes):
        values = {method: np.asarray(profiles[method].get(key, []), dtype=float)
                  for method in methods if method in profiles}
        valid = [value[np.isfinite(value)] for value in values.values() if np.any(np.isfinite(value))]
        if not valid:
            ax.set_axis_off(); continue
        pooled = np.concatenate(valid); lo, hi = np.percentile(pooled, [0.25, 99.5])
        span = max(float(hi - lo), .25); edges = np.linspace(max(0.0, float(lo - .06 * span)), float(hi + .06 * span), 45)
        centers = (edges[:-1] + edges[1:]) / 2
        for method, value in values.items():
            value = value[np.isfinite(value)]
            if not len(value):
                continue
            count, _ = np.histogram(np.clip(value, edges[0], edges[-1]), bins=edges)
            smooth = gaussian_filter1d(count.astype(float), sigma=1.1, mode="nearest")
            ax.plot(centers, smooth, color=COLORS[method], lw=1.65, label=PLOT_LABELS[method])
            ax.fill_between(centers, 0, smooth, color=COLORS[method], alpha=.045)
            rows.extend({"dataset": dataset, "group": group, "plane": code, "method": method,
                         "bin_left": float(left), "bin_right": float(right), "bin_center": float(center),
                         "hist_count": int(raw), "smoothed_count": float(sm), "sample_count": int(len(value))}
                        for left, right, center, raw, sm in zip(edges[:-1], edges[1:], centers, count, smooth))
        ax.set_title(label, fontweight="bold", fontsize=9.5); ax.set_xlabel("NIQE score"); ax.set_ylabel("Count")
        ax.grid(alpha=.17, linewidth=.5); ax.legend(loc="upper right", fontsize=6.2, frameon=False)
    fig.savefig(figures / f"niqe_curves_{dataset}.png", dpi=350); fig.savefig(figures / f"niqe_curves_{dataset}.pdf"); pdf.savefig(fig); plt.close(fig)
    return rows


def write_csv(path: Path, rows: Sequence[Dict[str, object]]) -> None:
    keys = sorted({key for row in rows for key in row})
    with path.open("w", newline="", encoding="utf-8-sig") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys)
        writer.writeheader(); writer.writerows(rows)


def select_high_fluctuation_window(stack, candidate_start: int, candidate_stop: int,
                                   requested_length: int) -> Tuple[int, int, Dict[str, object], List[Dict[str, object]]]:
    """Find one continuous held-out Z segment with the strongest stripe-like drift.

    The score operates on the high-pass component of the native-Z whole-slice
    mean.  It combines mean absolute adjacent change with the robust MAD of
    those changes.  Consequently gradual anatomical intensity drift is
    discounted, while persistent inter-slice brightness oscillation is kept.
    """
    if not (0 <= candidate_start < candidate_stop <= stack.count):
        raise ValueError(f"Invalid candidate range [{candidate_start}, {candidate_stop})")
    z = np.arange(candidate_start, candidate_stop, dtype=int)
    means = np.asarray([float(stack.read(int(index)).mean()) for index in z], dtype=float)
    available = len(z); length = min(max(2, requested_length), available)
    trend_sigma = max(3.0, min(15.0, length / 16.0))
    highpass = means - gaussian_filter1d(means, sigma=trend_sigma, mode="nearest")
    adjacent_abs = np.abs(np.diff(highpass))
    candidates: List[Dict[str, object]] = []
    for local_start in range(available - length + 1):
        changes = adjacent_abs[local_start:local_start + length - 1]
        center = float(np.median(changes)) if len(changes) else 0.0
        mad = float(np.median(np.abs(changes - center))) if len(changes) else 0.0
        candidates.append({
            "source_z_start": int(z[local_start]), "source_z_stop": int(z[local_start + length - 1] + 1),
            "length": int(length), "hf_mean_abs_adjacent_change": float(np.mean(changes)) if len(changes) else 0.0,
            "hf_mad_adjacent_change": mad, "hf_window_score": float(np.mean(changes) + mad) if len(changes) else 0.0,
        })
    best = max(candidates, key=lambda row: (float(row["hf_window_score"]), -int(row["source_z_start"])))
    analysis = {
        "candidate_z": z.astype(int).tolist(), "z_mean_intensity": means.tolist(),
        "z_mean_highpass": highpass.tolist(), "z_mean_highpass_abs_adjacent_diff": adjacent_abs.tolist(),
        "trend_sigma_slices": float(trend_sigma), **best,
    }
    return int(best["source_z_start"]), int(best["source_z_stop"]), analysis, candidates


def evaluation_window_page(dataset: str, analysis: Dict[str, object], figures: Path, pdf: PdfPages) -> None:
    """Persist the raw Z-axis evidence used to select the held-out benchmark window."""
    z = np.asarray(analysis["candidate_z"], dtype=int)
    means = np.asarray(analysis["z_mean_intensity"], dtype=float)
    highpass_diff = np.asarray(analysis["z_mean_highpass_abs_adjacent_diff"], dtype=float)
    start, stop = int(analysis["source_z_start"]), int(analysis["source_z_stop"])
    style(); fig, axes = plt.subplots(2, 1, figsize=(11.69, 5.2), sharex=True, constrained_layout=True)
    axes[0].plot(z, means, color="#4D4D4D", lw=1.1, label="Original global mean")
    axes[0].axvspan(start, stop - 1, color="#D55E00", alpha=.16, label="selected evaluation segment")
    axes[0].set_ylabel("Z mean intensity"); axes[0].set_title("Native-Z global brightness", fontweight="bold")
    axes[1].plot(z[1:], highpass_diff, color="#0072B2", lw=1.0, label="|Δ high-pass Z mean|")
    axes[1].axvspan(start, stop - 1, color="#D55E00", alpha=.16)
    axes[1].set_ylabel("High-pass adjacent change"); axes[1].set_xlabel("Source Z-slice index")
    axes[1].set_title("Stripe-sensitive fluctuation signal", fontweight="bold")
    for ax in axes:
        ax.grid(alpha=.2, linewidth=.55); ax.legend(frameon=False, fontsize=7.5, loc="upper right")
    fig.suptitle(f"{dataset}: automatically selected high-fluctuation held-out window Z={start}:{stop}", fontweight="bold")
    fig.savefig(figures / f"evaluation_window_selection_{dataset}.png", dpi=350)
    fig.savefig(figures / f"evaluation_window_selection_{dataset}.pdf")
    pdf.savefig(fig); plt.close(fig)


def read_qc(image: np.ndarray) -> Dict[str, float]:
    """Conservative normality descriptors for one raw XY image."""
    x = image.astype(np.float32)
    height, width = x.shape
    by, bx = max(1, height // 8), max(1, width // 8)
    blocks = [x[y:y + by, xx:xx + bx].mean() for y in range(0, height - by + 1, by)
              for xx in range(0, width - bx + 1, bx)]
    blocks = np.asarray(blocks, dtype=float)
    means = float(x.mean()); std = float(x.std())
    return {
        "mean": means, "std": std,
        "dark_fraction": float(np.mean(x <= 4)), "bright_fraction": float(np.mean(x >= 251)),
        "block_mean_cv": float(blocks.std() / max(abs(blocks.mean()), 1.0)),
        "block_mean_range": float(np.percentile(blocks, 95) - np.percentile(blocks, 5)),
        "border_dark_fraction": float(np.mean(np.concatenate((x[:8].ravel(), x[-8:].ravel(), x[:, :8].ravel(), x[:, -8:].ravel())) <= 4)),
    }


def robust_z(values: np.ndarray) -> np.ndarray:
    median = float(np.median(values)); mad = float(np.median(np.abs(values - median)))
    return (values - median) / max(1.4826 * mad, 1e-6)


def target_pool_size(count: int) -> int:
    """Reserve a contiguous prefix so real-data test Z remains uninterrupted."""
    return min(400, max(60, count // 4)) if count >= 120 else min(60, max(30, count // 2))


def target_selection_size(pool: int) -> int:
    """Enough images for stable NIQE statistics without mixing luminance modes.

    Each 1024x1024 image provides many overlapping NIQE patches.  Limiting a
    large raw pool to 120 selected slices is therefore statistically ample and
    permits a much narrower whole-slice brightness band than the old 200-image
    rule.
    """
    return min(120, max(20, pool // 2))


def quality_selection(stack, source_dataset: str, target_dataset: str, source_indices: Sequence[int],
                      target_count: int, target_dir: Path, refresh: bool,
                      require_exact_count: bool = False) -> Tuple[List[int], List[Dict[str, object]], Path]:
    """Select and preserve normal raw images without modifying source slices."""
    manifest = target_dir / "niqe_pristine_selection.json"
    selected_dir = target_dir / "niqe_pristine"
    if manifest.exists() and not refresh:
        data = json.loads(manifest.read_text(encoding="utf-8"))
        chosen = [int(i) for i in data["selected_source_indices"]]
        if (data.get("selection_schema") == PRISTINE_SELECTION_SCHEMA
                and all((selected_dir / item["saved_name"]).is_file() for item in data.get("saved_images", []))):
            return chosen, list(data["qc_rows"]), manifest
    rows: List[Dict[str, object]] = []
    for index in source_indices:
        row = {"source_dataset": source_dataset, "target_dataset": target_dataset, "source_z": int(index), **read_qc(stack.read(index))}
        rows.append(row)
    keys = ("mean", "std", "dark_fraction", "bright_fraction", "block_mean_cv", "block_mean_range", "border_dark_fraction")
    arrays = {key: np.asarray([row[key] for row in rows], dtype=float) for key in keys}
    q = {key: np.percentile(value, [5, 10, 90, 95]) for key, value in arrays.items()}
    score = (np.abs(robust_z(arrays["mean"])) + .55 * np.abs(robust_z(arrays["std"]))
             + .8 * np.maximum(robust_z(arrays["block_mean_cv"]), 0)
             + .8 * np.maximum(robust_z(arrays["block_mean_range"]), 0)
             + 3 * (arrays["dark_fraction"] + arrays["bright_fraction"] + arrays["border_dark_fraction"]))
    candidates: List[int] = []
    for pos, row in enumerate(rows):
        reasons: List[str] = []
        if row["std"] < max(4.0, q["std"][1]): reasons.append("low_intraslice_contrast")
        if row["dark_fraction"] > max(.04, q["dark_fraction"][2]): reasons.append("too_dark_or_missing")
        if row["bright_fraction"] > max(.04, q["bright_fraction"][2]): reasons.append("too_bright_or_saturated")
        if row["border_dark_fraction"] > max(.08, q["border_dark_fraction"][3]): reasons.append("dark_border")
        if row["block_mean_cv"] > q["block_mean_cv"][2] or row["block_mean_range"] > q["block_mean_range"][2]: reasons.append("brightness_nonuniform")
        if not (q["mean"][0] <= row["mean"] <= q["mean"][3]): reasons.append("brightness_outlier")
        row["quality_score"] = float(score[pos]); row["reasons"] = ";".join(reasons); row["accepted_candidate"] = not reasons
        if not reasons: candidates.append(pos)
    # Relax only the soft brightness-uniformity rule if conservative filters
    # leave too few candidates; missing/saturated images remain rejected.
    if len(candidates) < target_count:
        candidates = [i for i, row in enumerate(rows)
                      if "too_dark_or_missing" not in row["reasons"] and "too_bright_or_saturated" not in row["reasons"] and "dark_border" not in row["reasons"]]
        for i in candidates: rows[i]["fallback_relaxed_uniformity"] = True
    if len(candidates) < max(8, min(20, target_count)):
        raise RuntimeError(f"{target_dataset}: only {len(candidates)} normal NIQE training images passed review.")

    # The previous implementation drew one slice from every Z bin.  This
    # inadvertently forced bright and dark anatomical phases into one model.
    # Here we find the shortest interval on the *global mean-intensity axis*
    # containing the requested number of healthy candidates.  All selected
    # images consequently come from one compact, auditable luminance mode.
    requested = min(target_count, len(candidates))
    by_mean = sorted(candidates, key=lambda i: rows[i]["mean"])
    # For all ordinary volumes, lower the number of training slices only when
    # needed to meet a strict 15%-of-mean luminance-span criterion.  NIQE uses
    # patch statistics, so 15--20 normal 1024px images remain adequate.  An
    # exact count can be requested for a small fixed-volume protocol (SNEMI).
    min_requested = min(requested, 15)
    selected_count = requested
    threshold_relative = .15
    span_start = 0; compact_ok = False
    for count in range(requested, min_requested - 1, -1):
        start = min(range(len(by_mean) - count + 1),
                    key=lambda pos: rows[by_mean[pos + count - 1]]["mean"] - rows[by_mean[pos]]["mean"])
        lo, hi = rows[by_mean[start]]["mean"], rows[by_mean[start + count - 1]]["mean"]
        relative_span = (hi - lo) / max((hi + lo) / 2.0, 1e-6)
        if relative_span <= threshold_relative:
            span_start, selected_count, compact_ok = start, count, True
            break
        if count == requested:
            span_start = start
    if require_exact_count:
        selected_count = requested
        span_start = min(range(len(by_mean) - selected_count + 1),
                         key=lambda pos: rows[by_mean[pos + selected_count - 1]]["mean"] - rows[by_mean[pos]]["mean"])
        lo, hi = rows[by_mean[span_start]]["mean"], rows[by_mean[span_start + selected_count - 1]]["mean"]
        compact_ok = ((hi - lo) / max((hi + lo) / 2.0, 1e-6)) <= threshold_relative
    brightness_core = by_mean[span_start:span_start + selected_count]
    low_mean, high_mean = rows[brightness_core[0]]["mean"], rows[brightness_core[-1]]["mean"]
    relative_span = (high_mean - low_mean) / max((high_mean + low_mean) / 2.0, 1e-6)
    brightness_core_set = set(brightness_core)
    for pos, row in enumerate(rows):
        row["brightness_core_low"] = float(low_mean)
        row["brightness_core_high"] = float(high_mean)
        row["in_brightness_core"] = pos in brightness_core_set
        if row["accepted_candidate"] and pos not in brightness_core_set:
            row["reasons"] = ";".join(filter(None, [row["reasons"], "outside_compact_brightness_core"]))
    # The core has exactly the required cardinality.  Keep its source order so
    # copied files remain easy to inspect alongside the raw volume.
    chosen_positions: List[int] = sorted(brightness_core)
    chosen_positions.sort(); chosen = [int(source_indices[i]) for i in chosen_positions]
    selected_dir.mkdir(parents=True, exist_ok=True)
    keep_names = {f"pristine_{rank:03d}_source_z{int(source_indices[pos]):05d}.png"
                  for rank, pos in enumerate(chosen_positions)}
    # Remove stale auto-generated copies so the directory is an exact visual
    # record of the current manifest, rather than a mixture of old and new
    # selection rules.
    for stale in selected_dir.glob("pristine_*.png"):
        if stale.name not in keep_names:
            stale.unlink()
    saved = []
    source_paths = getattr(stack, "paths", None)
    for rank, pos in enumerate(chosen_positions):
        source_index = int(source_indices[pos]); name = f"pristine_{rank:03d}_source_z{source_index:05d}.png"; destination = selected_dir / name
        if not destination.exists() or refresh:
            if source_paths is not None:
                shutil.copy2(source_paths[source_index], destination)
            else:
                ok, encoded = cv2.imencode(".png", stack.read(source_index))
                if not ok: raise RuntimeError(f"Could not encode normal image {source_index}")
                destination.write_bytes(encoded.tobytes())
        rows[pos]["selected"] = True; saved.append({"source_z": source_index, "saved_name": name})
    for row in rows: row.setdefault("selected", False)
    payload = {
        "selection_schema": PRISTINE_SELECTION_SCHEMA,
        "source_dataset": source_dataset, "target_dataset": target_dataset,
        "selection_protocol": "automatic health audit + minimum-span compact global-brightness core; copied, never removed",
        "requested_count": int(target_count), "selected_count": len(chosen),
        "brightness_consistency_threshold_relative_span": threshold_relative,
        "compact_brightness_mean_range": [float(low_mean), float(high_mean)],
        "compact_brightness_relative_span": float(relative_span),
        "brightness_consistency_passed": bool(compact_ok),
        "selection_warning": ("Exact-count protocol retained despite raw-volume brightness drift; interpret this prior with the listed span."
                              if require_exact_count and not compact_ok else ""),
        "selected_source_indices": chosen, "saved_images": saved, "qc_rows": rows,
    }
    manifest.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    write_csv(target_dir / "niqe_pristine_qc.csv", rows)
    return chosen, rows, manifest


def selection_review_page(dataset: str, qc_rows: Sequence[Dict[str, object]], selected_dir: Path, pdf: PdfPages, figures: Path) -> None:
    style(); fig = plt.figure(figsize=(11.69, 8.27)); grid = fig.add_gridspec(2, 5, height_ratios=(1.05, 1.0))
    ax = fig.add_subplot(grid[0, :]); mean = np.asarray([r["mean"] for r in qc_rows]); std = np.asarray([r["std"] for r in qc_rows]); sel = np.asarray([bool(r.get("selected")) for r in qc_rows])
    ax.scatter(mean[~sel], std[~sel], s=14, c="#B8B8B8", alpha=.65, label="not selected")
    ax.scatter(mean[sel], std[sel], s=20, c="#D55E00", label="selected normal")
    ax.set_xlabel("Slice mean intensity"); ax.set_ylabel("Intra-slice standard deviation"); ax.set_title("Automatic normal-image audit", fontweight="bold"); ax.grid(alpha=.2); ax.legend(fontsize=8)
    images = sorted(selected_dir.glob("pristine_*.png"))[:5]
    for col in range(5):
        pane = fig.add_subplot(grid[1, col]); pane.axis("off")
        if col < len(images):
            raw = np.fromfile(str(images[col]), dtype=np.uint8); image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
            pane.imshow(image, cmap="gray", vmin=0, vmax=255); pane.set_title(images[col].stem.replace("pristine_", "#"), fontsize=7)
    fig.suptitle(f"{dataset}: NIQE pristine-image selection review", fontweight="bold")
    fig.savefig(figures / f"niqe_selection_review_{dataset}.png"); fig.savefig(figures / f"niqe_selection_review_{dataset}.pdf"); pdf.savefig(fig); plt.close(fig)


def coverage_page(rows: Sequence[Dict[str, object]], pdf: PdfPages, figures: Path) -> None:
    style(); cols = ["Dataset", "N", "Test start", "Test Z", "Normal source", "NIQE train", "Methods available", "Warnings"]
    def short_name(row: Dict[str, object]) -> str:
        name = str(row.get("dataset", ""))
        if name.startswith("jrc_mus-kidney_") and name.endswith(("-weak", "-medium", "-strong")):
            return f"kidney-sim-{name.rsplit('-', 1)[-1]}"
        return name if len(name) <= 34 else name[:31] + "..."
    body = [[short_name(r), r.get("count", ""), r.get("test_start", ""), r.get("test_z_count", ""), r.get("normal_source_short", r.get("normal_source", "")), r.get("niqe_train_count", ""), r.get("method_count", ""), r.get("warnings", "")] for r in rows]
    fig, ax = plt.subplots(figsize=(13.8, max(4.0, .38 * (len(body) + 2)))); ax.axis("off")
    tab = ax.table(cellText=body, colLabels=cols, cellLoc="center", loc="center", colWidths=[.20,.05,.07,.07,.14,.07,.25,.15]); tab.auto_set_font_size(False); tab.set_fontsize(6.8); tab.scale(1, 1.45)
    for c in range(len(cols)): tab[(0,c)].set_text_props(weight="bold")
    ax.set_title("Batch coverage, continuity split, and automatic NIQE-prior audit", fontweight="bold", pad=14)
    fig.savefig(figures / "coverage_and_protocol.png"); fig.savefig(figures / "coverage_and_protocol.pdf"); pdf.savefig(fig); plt.close(fig)


def audit_rows(dataset: str, rows: Sequence[Dict[str, object]]) -> List[Dict[str, object]]:
    report = []
    for row in rows:
        finite = {key: bool(np.isfinite(row.get(key, np.nan))) for key in ("niqe_xz", "niqe_yz", "brisque_xy", "piqe_xy", "asid_percent", "iscc")}
        report.append({"dataset": dataset, "method": row["method"], "all_core_metrics_finite": all(finite.values()),
                       "nonfinite_metrics": ";".join(key for key, ok in finite.items() if not ok),
                       "test_z_is_continuous": True, "global_matching_applied": bool(row.get("global_matching_applied", False)),
                       "matching_target": row.get("matching_target", ""),
                       "result_cache_hit": bool(row.get("result_cache_hit", False))})
    return report


def compute_or_load_method(dataset: str, method: str, stack, eval_gt, mu: np.ndarray, cov: np.ndarray,
                           model_path: Path, z_ratio: float, args, brisque, cache_root: Path,
                           cache_lookup_roots: Sequence[Path], niqe_cache: Path, test_start: int,
                           score_protocol: Dict[str, object]) -> Tuple[Dict[str, object], Dict[str, object], bool, str]:
    """Evaluate one already saved method once; subsequently reuse its exact cache."""
    key = method_cache_key(dataset, method, stack, model_path, args, z_ratio, score_protocol)
    for lookup_root in cache_lookup_roots:
        loaded = load_method_cache(lookup_root, key)
        if loaded is not None:
            metrics, profile = loaded
            return metrics, profile, True, key
    metrics, profile = nonreference(
        stack, mu, cov, 96, z_ratio, 0, 1, args.metric_cross_sections, args.iqa_slices,
        brisque, args.cross_plane_border, 0, 1,
        niqe_cache / f"{dataset}_{method}_xy_metric_{key[:12]}.npz", 20260721,
    )
    xy_count = stack.count if args.niqe_xy_slices <= 0 else min(args.niqe_xy_slices, stack.count)
    xy_ids = spaced(stack.count, xy_count)
    xy_cache = niqe_cache / f"{dataset}_{method}_xy_n{xy_count}_{key[:12]}.npz"
    xy_scores: List[float] = []
    if xy_cache.exists():
        try:
            saved = np.load(xy_cache)
            if saved["indices"].astype(int).tolist() == xy_ids:
                xy_scores = saved["scores"].astype(float).tolist()
        except (OSError, KeyError, ValueError):
            xy_scores = []
    if len(xy_scores) != len(xy_ids):
        xy_scores = []
        for begin in range(0, len(xy_ids), args.gpu_batch_size):
            batch = [stack.read(index) for index in xy_ids[begin:begin + args.gpu_batch_size]]
            xy_scores.extend(niqe_scores_gpu(batch, mu, cov, 96, batch_size=args.gpu_batch_size))
        np.savez_compressed(xy_cache, indices=np.asarray(xy_ids), scores=np.asarray(xy_scores, dtype=np.float64))
    xy_scores = [float(value) for value in xy_scores if np.isfinite(value)]
    metrics["niqe_xy"] = float(np.mean(xy_scores)) if xy_scores else np.nan
    metrics["niqe_xy_sample_count"] = len(xy_scores)
    profile["niqe_xy_samples"] = xy_scores
    profile["niqe_xy_z_indices"] = [test_start + int(index) for index in xy_ids]

    volume = load(stack, list(range(stack.count)))
    count_tag = args.niqe_cross_sections if args.niqe_cross_sections > 0 else "all"
    tag = f"{dataset}_{method}_z{test_start}-{test_start + stack.count}_border{args.cross_plane_border}_n{count_tag}_{key[:12]}"
    _, xz = cached_axis_scores(volume, "xz", mu, cov, 96, z_ratio, args.cross_plane_border, 1,
                               niqe_cache / f"{tag}_xz.npz", 1, "gpu", args.gpu_batch_size, args.niqe_cross_sections)
    _, yz = cached_axis_scores(volume, "yz", mu, cov, 96, z_ratio, args.cross_plane_border, 1,
                               niqe_cache / f"{tag}_yz.npz", 1, "gpu", args.gpu_batch_size, args.niqe_cross_sections)
    del volume
    profile["niqe_xz_samples"] = [float(value) for value in xz if np.isfinite(value)]
    profile["niqe_yz_samples"] = [float(value) for value in yz if np.isfinite(value)]
    metrics["niqe_xz"] = float(np.mean(profile["niqe_xz_samples"])) if profile["niqe_xz_samples"] else np.nan
    metrics["niqe_yz"] = float(np.mean(profile["niqe_yz_samples"])) if profile["niqe_yz_samples"] else np.nan
    if eval_gt is not None:
        metrics.update(full_reference(stack, eval_gt, samples=min(200, stack.count), start=0))
    else:
        metrics.update({"psnr": np.nan, "ssim": np.nan, "mae": np.nan, "msssim": np.nan, "edge_corr": np.nan})
    profile["profile_z"] = [test_start + int(value) for value in profile["profile_z"]]
    provenance = {"cache_schema": METHOD_CACHE_SCHEMA, "dataset": dataset, "method": method,
                  "score_protocol": score_protocol, "test_start": test_start, "test_count": stack.count,
                  "source": stack_fingerprint(stack), "model": str(model_path)}
    save_method_cache(cache_root, key, metrics, profile, provenance)
    return metrics, profile, False, key


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    parser = argparse.ArgumentParser(description="Reusable all-dataset EM restoration benchmark")
    parser.add_argument("--output-dir", type=Path, default=Path("zstripe_benchmark_saved_outputs"))
    parser.add_argument("--reuse-cache-from", type=Path, default=None,
                        help="Optional completed benchmark directory used read-only for NIQE priors and method caches.")
    parser.add_argument("--sscor-root", type=Path, default=root.parent / "SSCOR-master" / "vem_sscor_runs",
                        help="Optional SSCOR output root; only matching complete restored stacks are evaluated.")
    parser.add_argument("--datasets", default="", help="Comma-separated subset; empty processes every valid test_data dataset.")
    parser.add_argument("--refresh-pristine-selection", action="store_true")
    parser.add_argument("--prepare-pristine-only", action="store_true",
                        help="Refresh/audit real-data NIQE-prior images and manifests, then exit without scoring methods.")
    parser.add_argument("--select-evaluation-windows-only", action="store_true",
                        help="Analyse Original Z-mean curves, save the high-fluctuation continuous test windows, then exit without NIQE scoring.")
    parser.add_argument("--gpu-batch-size", type=int, default=8)
    parser.add_argument("--iqa-slices", type=int, default=8)
    parser.add_argument("--metric-cross-sections", type=int, default=32)
    parser.add_argument("--niqe-cross-sections", type=int, default=200,
                        help="Uniformly sampled interior XZ/YZ planes per axis for NIQE curves (0: every plane).")
    parser.add_argument("--niqe-xy-slices", type=int, default=200,
                        help="Uniformly sampled held-out XY slices per method for GPU NIQE (0: all held-out slices).")
    parser.add_argument("--cross-plane-border", type=int, default=32)
    parser.add_argument("--display-mode", choices=("raw",), default="raw")
    parser.add_argument("--test-z-length", type=int, default=200,
                        help="Length of the continuous held-out Z segment (0: use every remaining slice).")
    parser.add_argument("--evaluation-window-mode", choices=("tail", "max-hf-mad"), default="max-hf-mad",
                        help="For real volumes, use the tail or select the held-out segment with maximum robust high-frequency Z-mean fluctuation.")
    args = parser.parse_args()
    output = args.output_dir if args.output_dir.is_absolute() else root / args.output_dir
    output.mkdir(parents=True, exist_ok=True); figures = output / "figures"; figures.mkdir(exist_ok=True); tables = output / "tables"; tables.mkdir(exist_ok=True)
    cache = output / "niqe_cache"; cache.mkdir(exist_ok=True); method_cache = output / "method_result_cache"; method_cache.mkdir(exist_ok=True)
    reuse_root = args.reuse_cache_from.resolve() if args.reuse_cache_from is not None else None
    if reuse_root is not None and not reuse_root.is_dir():
        parser.error(f"--reuse-cache-from does not exist: {reuse_root}")
    reuse_niqe = reuse_root / "niqe_cache" if reuse_root is not None else None
    cache_lookup_roots = [method_cache] + ([reuse_root / "method_result_cache"] if reuse_root is not None else [])
    test_root = configured_root(root, "VEMDESTRIPE_TEST_DATA_ROOT", "test_data")
    names = {p.name for p in test_root.iterdir() if p.is_dir()}
    simulation_root = configured_root(root, "VEMDESTRIPE_SIM_DATA_ROOT", "testdata-sim")
    if simulation_root.is_dir(): names.update(p.name for p in simulation_root.iterdir() if p.is_dir())
    requested = {x.strip() for x in args.datasets.split(",") if x.strip()}
    names = sorted(requested) if requested else sorted(names)
    if args.prepare_pristine_only:
        selection_audit_rows: List[Dict[str, object]] = []
        for dataset in names:
            if dataset in SIMULATED:
                continue
            stacks = stacks_for_dataset(root, dataset, args.sscor_root); original = stacks.get("Original")
            if original is None:
                print(f"[skip] {dataset}: Original stack unavailable", flush=True); continue
            # SNEMItrain contains 100 raw slices.  Reserve a 40-slice
            # continuous tail for testing and select at most 20 compact-
            # brightness NIQE-prior slices from the first 60 of SNEMItrain.
            if dataset == "snemitrain":
                pool, wanted = min(60, original.count - 2), 20
            else:
                pool, wanted = target_pool_size(original.count), target_selection_size(target_pool_size(original.count))
            if pool <= 0:
                print(f"[skip] {dataset}: insufficient raw slices", flush=True); continue
            folder = test_root / dataset
            chosen, qc_rows, manifest = quality_selection(original, dataset, dataset, list(range(pool)), wanted, folder,
                                                            args.refresh_pristine_selection)
            selected = [row for row in qc_rows if row.get("selected")]
            mean_values = np.asarray([row["mean"] for row in selected], dtype=float)
            selection_audit_rows.append({"dataset": dataset, "source_dataset": dataset, "pool_size": pool,
                               "selected_count": len(chosen), "test_start": pool,
                               "mean_min": float(mean_values.min()), "mean_max": float(mean_values.max()),
                               "mean_span": float(np.ptp(mean_values)), "manifest": str(manifest)})
            print(f"[selected] {dataset}: {len(chosen)} images, mean range {mean_values.min():.3f}-{mean_values.max():.3f}", flush=True)
        write_csv(output / "pristine_selection_refresh_summary.csv", selection_audit_rows)
        (output / "pristine_selection_refresh_summary.json").write_text(json.dumps(selection_audit_rows, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"Saved selection audit: {output / 'pristine_selection_refresh_summary.csv'}", flush=True)
        return
    if args.select_evaluation_windows_only:
        rows: List[Dict[str, object]] = []
        report = output / "evaluation_window_selection_report.pdf"
        with PdfPages(report) as pdf:
            for dataset in names:
                if dataset in SIMULATED:
                    continue
                original = stacks_for_dataset(root, dataset, args.sscor_root).get("Original")
                if original is None:
                    print(f"[skip] {dataset}: Original stack unavailable", flush=True); continue
                pool = min(60, original.count - 2) if dataset == "snemitrain" else target_pool_size(original.count)
                requested = original.count - pool if args.test_z_length <= 0 else args.test_z_length
                start, stop, analysis, candidates = select_high_fluctuation_window(original, pool, original.count, requested)
                evaluation_window_page(dataset, analysis, figures, pdf)
                rows.append({"dataset": dataset, "candidate_start": pool, "candidate_stop": original.count,
                             "test_z_start": start, "test_z_stop": stop, "test_z_count": stop - start,
                             "hf_window_score": analysis["hf_window_score"],
                             "hf_mean_abs_adjacent_change": analysis["hf_mean_abs_adjacent_change"],
                             "hf_mad_adjacent_change": analysis["hf_mad_adjacent_change"],
                             "trend_sigma_slices": analysis["trend_sigma_slices"]})
                write_csv(tables / f"{dataset}_raw_z_intensity_selection.csv",
                          [{"dataset": dataset, "source_z": int(z), "z_mean_intensity": float(mean), "z_mean_highpass": float(hp)}
                           for z, mean, hp in zip(analysis["candidate_z"], analysis["z_mean_intensity"], analysis["z_mean_highpass"])])
                write_csv(tables / f"{dataset}_evaluation_window_candidates.csv", candidates)
                print(f"[window] {dataset}: Z={start}:{stop}, score={analysis['hf_window_score']:.4f}", flush=True)
        write_csv(output / "evaluation_window_selection_summary.csv", rows)
        print(f"Saved evaluation-window analysis: {report}", flush=True)
        return
    if torch is None or not torch.cuda.is_available(): raise RuntimeError("This batch workflow requires CUDA PyTorch in stripe_removal.")
    brisque = make_brisque(); all_metrics: List[Dict[str, object]] = []; all_niqe: List[Dict[str, object]] = []; curve_bins: List[Dict[str, object]] = []
    all_qc: List[Dict[str, object]] = []; coverage: List[Dict[str, object]] = []; all_audit: List[Dict[str, object]] = []; input_manifest: List[Dict[str, object]] = []
    metadata: Dict[str, object] = {
        "arguments": vars(args), "engine": "CUDA NIQE features + CPU covariance/IQA", "method_result_cache_schema": METHOD_CACHE_SCHEMA,
        "protocol": {
            "simulated": "Score saved final outputs unchanged; clean TIFF is used only for paired metrics and declared NIQE priors.",
            "real": "Score saved final outputs unchanged. vEMDestripe calibration is completed and saved upstream; delivered baselines are not calibrated here.",
            "niqe": "Per-dataset pristine model; 200 held-out XY samples and 200 uniformly spaced interior XZ/YZ samples by default; raw scores and curve bins are saved.",
        }, "datasets": {},
    }
    # A multi-hour run can be interrupted after several dataset pages have
    # already been written.  PdfPages only writes its cross-reference table
    # when it closes, so writing directly to the public report path leaves an
    # unreadable PDF after an interruption.  Keep the work-in-progress file
    # separate and publish the report only after every data artifact is saved.
    report_path = output / "zstripe_all_datasets_report.pdf"
    inprogress_report = output / "zstripe_all_datasets_report.inprogress.pdf"
    with PdfPages(inprogress_report) as pdf:
        for dataset in names:
            print(f"[dataset] {dataset}", flush=True)
            stacks = stacks_for_dataset(root, dataset, args.sscor_root); original = stacks.get("Original")
            if original is None:
                coverage.append({"dataset": dataset, "count": 0, "test_start": "-", "test_z_count": "-", "normal_source": "-", "normal_source_short":"-", "niqe_train_count": "-", "method_count": "-", "methods": "-", "warnings": "Original or aligned result stack unavailable"})
                continue
            gt = ground_truth(root, dataset); target_folder = test_root / dataset
            if dataset in SIMULATED:
                if gt is None: raise RuntimeError(f"{dataset}: registered clean TIFF is unavailable")
                train_source, train_indices, test_start, qc_rows = gt, list(range(min(200, gt.count))), 200, []
                normal_source = "clean paired TIFF (first 200)"; manifest = None
                mu_path = cache / f"{dataset}_clean_pristine_200_patch96.npz"
                if reuse_niqe is not None and not mu_path.exists() and (reuse_niqe / mu_path.name).is_file():
                    mu_path = reuse_niqe / mu_path.name
                if mu_path.exists():
                    saved = np.load(mu_path); mu, cov, patches = saved["mu"], saved["cov"], int(saved["patches"])
                else:
                    mu, cov, patches = fit_pristine_model((gt.read(i) for i in train_indices), 96, .75); np.savez_compressed(mu_path, mu=mu, cov=cov, patches=np.asarray(patches))
            else:
                donor = SNEMI_DONOR.get(dataset, dataset)
                donor_stack = stacks_for_dataset(root, donor, args.sscor_root).get("Original") if donor != dataset else original
                if donor_stack is None: raise RuntimeError(f"{dataset}: normal-image donor {donor} is unavailable")
                if dataset == "snemitrain":
                    # Same-domain NIQE prior: the first 60 SNEMItrain slices
                    # are audited and at most 20 compact-brightness slices
                    # train the prior; [60, 100) remains never-seen test Z.
                    pool, wanted = min(60, original.count - 2), 20
                else:
                    pool = donor_stack.count if donor != dataset else target_pool_size(original.count)
                    wanted = target_selection_size(pool)
                source_indices = list(range(pool))
                train_indices, qc_rows, manifest = quality_selection(
                    donor_stack, donor, dataset, source_indices, wanted, target_folder,
                    args.refresh_pristine_selection)
                train_source = donor_stack; test_start = 0 if donor != dataset else pool; normal_source = f"{donor} raw normal audit"
                # The selected images, not merely their number, define a NIQE
                # prior.  Include their immutable source-index signature so a
                # refresh can never silently reuse an obsolete model.
                prior_tag = stable_hash({"schema": PRISTINE_SELECTION_SCHEMA, "source": donor, "indices": train_indices})[:12]
                mu_path = cache / f"{dataset}_raw_pristine_{len(train_indices)}_{prior_tag}_patch96.npz"
                if reuse_niqe is not None and not mu_path.exists() and (reuse_niqe / mu_path.name).is_file():
                    mu_path = reuse_niqe / mu_path.name
                if mu_path.exists():
                    saved = np.load(mu_path); mu, cov, patches = saved["mu"], saved["cov"], int(saved["patches"])
                else:
                    mu, cov, patches = fit_pristine_model((train_source.read(i) for i in train_indices), 96, .75); np.savez_compressed(mu_path, mu=mu, cov=cov, patches=np.asarray(patches))
                all_qc.extend(qc_rows)
            if test_start >= original.count - 2: raise RuntimeError(f"{dataset}: normal pool leaves too few continuous test Z slices")
            evaluation_window: Optional[Dict[str, object]] = None
            evaluation_window_candidates: List[Dict[str, object]] = []
            if dataset not in SIMULATED and args.evaluation_window_mode == "max-hf-mad":
                requested = original.count - test_start if args.test_z_length <= 0 else args.test_z_length
                test_start, test_stop, evaluation_window, evaluation_window_candidates = select_high_fluctuation_window(
                    original, test_start, original.count, requested)
            else:
                test_stop = original.count if args.test_z_length <= 0 else min(original.count, test_start + args.test_z_length)
            if test_stop - test_start < 2:
                raise RuntimeError(f"{dataset}: held-out Z window [{test_start}, {test_stop}) is too short")
            # A method may be an explicitly validated aligned prefix (the
            # Bumbarger EMISAC 2000-Z output).  Retain it only when the whole
            # selected continuous evaluation window is covered.
            available_methods = [method for method in METHODS if stacks.get(method) is not None and stacks[method].count >= test_stop]
            eval_stacks, eval_gt = prepare_saved_outputs(
                stacks, available_methods, test_start, test_stop, gt if dataset in SIMULATED else None)
            methods = [method for method in METHODS if eval_stacks.get(method) is not None]
            ratio = 6.0 if uses_z6_resampling(dataset) else 1.0
            coverage.append({"dataset": dataset, "count": original.count, "test_start": test_start, "test_z_count": test_stop-test_start, "normal_source": normal_source, "normal_source_short": "clean paired TIFF" if dataset in SIMULATED else (f"{donor} normal audit"), "niqe_train_count": len(train_indices), "method_count": len(methods), "methods": ";".join(methods), "warnings": ";".join(method for method in METHODS if stacks.get(method) is None)})
            profiles: Dict[str, Dict[str, object]] = {}; dataset_rows: List[Dict[str, object]] = []
            for method in methods:
                stack = eval_stacks[method]
                entry = {"evaluator_photometric_transform": False, "input_pixels": "saved_final_outputs"}
                base_metrics, profile, cache_hit, cache_key = compute_or_load_method(dataset, method, stack, eval_gt, mu, cov, mu_path, ratio, args, brisque, method_cache, cache_lookup_roots, cache, test_start, entry)
                print(f"  [method] {method} {'cache' if cache_hit else 'computed'}", flush=True)
                metrics = dict(base_metrics)
                metrics.update({"dataset":dataset,"method":method,"test_z_start":test_start,"test_z_stop":test_stop,"test_z_count":stack.count,
                                "z_resampled_for_xz_yz":ratio,"cross_plane_inplane_border":args.cross_plane_border,
                                "niqe_training_source":normal_source,"niqe_training_count":len(train_indices),
                                "niqe_cross_sections_per_axis":min(args.niqe_cross_sections, 1024 - 2 * args.cross_plane_border) if args.niqe_cross_sections > 0 else 1024 - 2 * args.cross_plane_border,
                                "global_matching_applied":False,"evaluation_input":"saved_final_outputs",
                                "result_cache_hit":cache_hit,"result_cache_key":cache_key})
                profiles[method] = profile; dataset_rows.append(metrics); all_metrics.append(metrics)
                input_manifest.append({"dataset":dataset,"method":method,"source_signature":stable_hash(stack_fingerprint(stack)),"result_cache_key":cache_key,
                                       "evaluator_photometric_transform":False,"test_z_start":test_start,"test_z_stop":test_stop})
            if evaluation_window is not None:
                evaluation_window_page(dataset, evaluation_window, figures, pdf)
                raw_rows = [{"dataset": dataset, "source_z": int(z), "z_mean_intensity": float(mean),
                             "z_mean_highpass": float(hp)}
                            for z, mean, hp in zip(evaluation_window["candidate_z"], evaluation_window["z_mean_intensity"], evaluation_window["z_mean_highpass"])]
                write_csv(tables / f"{dataset}_raw_z_intensity_selection.csv", raw_rows)
                write_csv(tables / f"{dataset}_evaluation_window_candidates.csv", evaluation_window_candidates)
            if qc_rows: selection_review_page(dataset, qc_rows, target_folder / "niqe_pristine", pdf, figures)
            comparison_page(dataset, eval_stacks, methods, ratio, args.cross_plane_border, figures, pdf, args.display_mode, 0, "_saved_outputs", f"saved final outputs; no evaluator normalization; source Z {test_start}:{test_stop}")
            table_page(dataset, dataset_rows, figures, pdf); specialized_table_page(dataset, dataset_rows, figures, pdf)
            curves_page(dataset, profiles, methods, figures, pdf, "_saved_outputs")
            curve_bins.extend(paper_niqe_curve_page(dataset, profiles, methods, figures, pdf, "saved_outputs"))
            for method, profile in profiles.items():
                for plane, key in (("xy", "niqe_xy_samples"), ("xz", "niqe_xz_samples"), ("yz", "niqe_yz_samples")):
                    all_niqe.extend({"dataset":dataset,"method":method,"plane":plane,"niqe_score":float(score),"evaluator_photometric_transform":False,
                                     "test_z_start":test_start,"test_z_stop":test_stop} for score in profile.get(key, []) if np.isfinite(score))
            write_csv(tables / f"{dataset}_metrics.csv", dataset_rows); all_audit.extend(audit_rows(dataset, dataset_rows))
            metadata["datasets"][dataset] = {"test_z_start":test_start,"test_z_stop":test_stop,"test_z_count":test_stop-test_start,"normal_source":normal_source,"niqe_train_indices":train_indices,"selection_manifest":str(manifest) if manifest else None,"niqe_pristine_patches":patches,"z_ratio":ratio,"methods":methods,"evaluator_photometric_transform":False,"evaluation_window_mode":args.evaluation_window_mode,"evaluation_window_analysis":evaluation_window}
        coverage_page(coverage, pdf, figures)
    write_csv(output / "metrics_all.csv", all_metrics); write_csv(output / "niqe_score_samples.csv", all_niqe); write_csv(output / "niqe_curve_bins.csv", curve_bins); write_csv(output / "niqe_pristine_qc_all.csv", all_qc); write_csv(output / "coverage.csv", coverage); write_csv(output / "result_audit.csv", all_audit); write_csv(output / "input_method_manifest.csv", input_manifest)
    (output / "metadata.json").write_text(json.dumps(metadata, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    inprogress_report.replace(report_path)
    print(f"Finished all-dataset benchmark report: {report_path}")


if __name__ == "__main__":
    main()
