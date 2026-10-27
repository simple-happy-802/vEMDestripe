#!/usr/bin/env python3
"""Author-side packaging of the complete, reference-calibrated vEMDestripe demo.

The full original output volume is read, not just the displayed 32 sections.
All baseline images and original EMISAC provenance are copied unchanged from
the verified v2 bundle. Only vEMDestripe receives the explicitly recorded
whole-volume reference mean/std calibration. Private source paths are never
written into the release archive.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor
import csv
import hashlib
import json
import re
import shutil
import zipfile
from pathlib import Path

import numpy as np
from PIL import Image

BUNDLE_NAME = "vemdestripe_bumbarger_reference_pipeline_v3"
DATASET_ID = "bumbarger138092_8092_4096_4096tile_r03_c03_1024_stack"
SOURCE_LABEL = "E1_block16"  # Private development label, not the public method name.
CHECKPOINT_SHA256 = "9a01cdef1f09716fe87369b057633f9e1278e86e77cbc70eb66b75798709c44a"


def sha256(path: Path) -> str:
    result = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            result.update(block)
    return result.hexdigest()


def write_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
                    encoding="utf-8")


def read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def moments(hist: np.ndarray) -> tuple[float, float]:
    # The exact population-moment expression used by the supplied common.py.
    p = hist.mean(axis=0)
    levels = np.arange(256, dtype=float)
    mu = float(p @ levels)
    return mu, float(np.sqrt(max(0, p @ (levels * levels) - mu * mu)))


def affine(source: tuple[float, float], target: tuple[float, float]) -> tuple[float, float]:
    mu, sd = source
    tm, ts = target
    if sd <= 1e-8:
        if ts > 1e-8:
            raise ValueError("Cannot match a constant volume")
        return 1., tm - mu
    scale = ts / sd
    return scale, tm - scale * mu


def lookup_table(scale: float, bias: float) -> np.ndarray:
    # Preserve float32 grey levels, clip-then-rint order and uint8 conversion.
    return np.rint(np.clip(np.arange(256, dtype=np.float32) * scale + bias, 0, 255)).astype(np.uint8)


def read_u8(path: Path) -> np.ndarray:
    with Image.open(path) as image:
        array = np.asarray(image)
    if array.dtype != np.uint8 or array.shape != (1024, 1024):
        raise ValueError(f"Expected a 1024-square uint8 grayscale section: {path.name}")
    return array


def measure(path: Path) -> tuple[np.ndarray, dict]:
    # Each image is read once for its cryptographic hash and once for decoding.
    array = read_u8(path)
    return np.bincount(array.ravel(), minlength=256) / array.size, {
        "name": path.name, "sha256": sha256(path), "pixels": int(array.size),
    }


def verify_base(root: Path) -> dict:
    manifest = read_json(root / "manifest.json")
    if manifest.get("schema_version") != 2 or manifest.get("input_local_z") != [777, 809]:
        raise ValueError("Expected the independently verified original-EMISAC v2 bundle")
    for name, expected in manifest["files"].items():
        if sha256(root / name) != expected:
            raise ValueError(f"Base-bundle checksum mismatch: {name}")
    return manifest


def build(args: argparse.Namespace) -> tuple[Path, str]:
    base, comparison, output = args.base_bundle.resolve(), args.comparison_root.resolve(), args.output_dir.resolve()
    archive = output.with_suffix(".zip")
    if output.name != BUNDLE_NAME:
        raise ValueError(f"The release directory must be named {BUNDLE_NAME}")
    if output.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite an existing bundle or archive")
    if args.workers < 1 or args.workers > 16:
        raise ValueError("Use between 1 and 16 input-reading workers")
    base_manifest = verify_base(base)
    plan_path = comparison / "plan.json"
    plan = read_json(plan_path)
    dataset = next(d for d in plan["datasets"] if d["dataset"] == DATASET_ID)
    source = dataset["sources"][SOURCE_LABEL]
    refs = dataset["reference_calibration"][SOURCE_LABEL]
    if dataset["count"] != 2651 or dataset["shape"] != [1024, 1024] or source["count"] != 2651:
        raise ValueError("Expected the full 2651-section Bumbarger output")
    files = [Path(name) for name in source["files"]]
    expected_names = [f"slice_{z:05d}_deflickered.png" for z in range(2651)]
    if [p.name for p in files] != expected_names:
        raise ValueError("The saved source must cover every original output section exactly once")
    # Recheck the immutable source inventory used in the audited comparison.
    fingerprint = hashlib.sha256()
    for path in files:
        stat = path.stat()
        fingerprint.update(f"{path.resolve()}|{stat.st_size}|{stat.st_mtime_ns}\n".encode())
    if fingerprint.hexdigest() != source["fingerprint"]:
        raise ValueError("The original full output source changed after the audited comparison")
    info_path = Path(refs["metadata_path"])
    if sha256(info_path) != refs["metadata_sha256"]:
        raise ValueError("Original reference-selection metadata changed")
    info = read_json(info_path)
    names = info["reference_indices"]
    if (info["num_input_images_total"] != 2651 or info["num_reference_used"] != 11
            or info["reference_structure_destruction"] != "mean_fusion_then_block_mean_16"):
        raise ValueError("The frozen result is not the selected complete 16x16 pipeline")
    if sorted(p.name for p in (base / "reference").glob("*.png")) != sorted(names):
        raise ValueError("The supplied bundle reference pool is not the original selected pool")
    reference_inventory = []
    reference_counts = np.zeros(256, np.float64)
    for record in refs["files"]:
        original = Path(record["path"])
        bundled = base / "reference" / original.name
        if sha256(original) != record["sha256"] or sha256(bundled) != record["sha256"]:
            raise ValueError(f"Original reference bytes differ: {original.name}")
        image = read_u8(bundled)
        reference_counts += np.bincount(image.ravel(), minlength=256)
        reference_inventory.append({"path": "reference/" + original.name,
                                    "sha256": record["sha256"], "pixels": int(image.size)})
    reference_histogram = reference_counts / reference_counts.sum()
    target_moments = moments(reference_histogram[None, :])
    if not np.allclose(target_moments, (refs["mean"], refs["std"]), atol=1e-12, rtol=0):
        raise ValueError("Reference pooled moments disagree with the evaluated pipeline")
    for z in range(777, 809):
        if sha256(files[z]) != sha256(base / "precomputed/vEMDestripe" / f"slice_{z:05d}.png"):
            raise ValueError(f"Frozen demo source is not the corresponding original output: Z={z}")
    print("Source identity and all 11 original reference snapshots verified", flush=True)
    print("Recomputing all 2651 output histograms, not only the demo window", flush=True)
    histograms, inventory = [], []
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        for index, (histogram, record) in enumerate(pool.map(measure, files), 1):
            histograms.append(histogram)
            inventory.append(record)
            if index % 400 == 0 or index == len(files):
                print(f"Measured {index}/{len(files)} complete output sections", flush=True)
    histograms = np.stack(histograms)
    cached_hist_path = comparison / "results" / DATASET_ID / SOURCE_LABEL / "raw_full_histograms.npz"
    with np.load(cached_hist_path) as cached:
        if not np.array_equal(histograms, cached["histograms"]):
            raise ValueError("Fresh full-volume histograms differ from the audited source")
    source_moments = moments(histograms)
    scale, bias = affine(source_moments, target_moments)
    lut = lookup_table(scale, bias)
    with (comparison / "calibration_parameters.csv").open(encoding="utf-8-sig", newline="") as handle:
        row = next(r for r in csv.DictReader(handle)
                   if r["dataset"] == DATASET_ID and r["method"] == SOURCE_LABEL)
    expected = tuple(float(row[key]) for key in
                     ("output_mean_before", "output_std_before", "target_mean", "target_std", "affine_scale", "affine_bias"))
    current = (*source_moments, *target_moments, scale, bias)
    if not np.allclose(current, expected, atol=1e-12, rtol=0):
        raise ValueError("Fresh calibration parameters differ from the provided evaluation CSV")
    if not np.array_equal(lut, lookup_table(expected[-2], expected[-1])):
        raise ValueError("Calibration mapping differs from the evaluated 256-level LUT")
    checkpoint = Path(__file__).resolve().parents[1] / "checkpoints/vemdestripe_seed42/best.pth"
    if sha256(checkpoint) != CHECKPOINT_SHA256:
        raise ValueError("Unexpected public checkpoint")
    calibration_path = comparison / "results" / DATASET_ID / SOURCE_LABEL / "CALIBRATION.json"
    marker_path = calibration_path.with_name("COMPLETE.json")
    marker = read_json(marker_path)
    if not marker.get("complete") or not marker.get("pipeline_verified"):
        raise ValueError("The original pipeline evaluation was not complete")

    shutil.copytree(base, output)
    calibration_folder = output / "calibration"
    calibration_folder.mkdir()
    hist_destination = calibration_folder / "source_full_histograms.npz"
    np.savez_compressed(hist_destination, histograms=histograms)
    inventory_destination = calibration_folder / "source_output_inventory.json"
    write_json(inventory_destination, {"section_count": len(inventory), "first_id": 0,
                                      "scope": "all network-output PNG files before calibration", "files": inventory})
    public_info = {key: value for key, value in info.items() if key != "reference_path"}
    public_info["reference_path"] = "reference/ (original unprocessed snapshots; fused/block image not used as target)"
    write_json(calibration_folder / "original_processing_info.json", public_info)
    derivation = {
        "source_development_label": SOURCE_LABEL,
        "public_method": "vEMDestripe",
        "comparison_protocol": plan["protocol"],
        "plan_sha256": sha256(plan_path),
        "calibration_csv_sha256": sha256(comparison / "calibration_parameters.csv"),
        "original_calibration_record_sha256": sha256(calibration_path),
        "original_evaluation_marker_sha256": sha256(marker_path),
        "original_processing_info_sha256": sha256(info_path),
        "source_fingerprint_verified": True,
        "source_fingerprint_note": "Original author-side frozen inventory was checked; portable file SHA256 inventory supplied separately",
        "fresh_full_source_histograms_equal_audited_histograms": True,
        "all_32_network_raw_demo_pngs_equal_full_source_files": True,
        "all_11_reference_pngs_equal_metadata_selected_original_snapshots": True,
        "fresh_parameters_equal_provided_calibration_csv": True,
        "rounding_contract": "np.rint(np.clip(np.arange(256,dtype=np.float32)*scale+bias,0,255)).astype(np.uint8)",
        "supplied_code_sha256": {Path(path).name: digest for path, digest in plan["code"].items()
                                  if Path(path).name in ("common.py", "evaluate_reference_pipeline.py", "run.py")},
    }
    write_json(calibration_folder / "derivation.json", derivation)
    record = {
        "schema_version": 1,
        "protocol": "vemdestripe_full_volume_reference_calibration_v1",
        "method": "vEMDestripe",
        "only_method_calibrated": "vEMDestripe",
        "source_scope": "all_saved_output_voxels",
        "target_source": "pooled_original_reference_snapshots",
        "source_statistics": {"section_count": 2651, "pixel_count": 2651 * 1024 * 1024,
                              "shape_yx": [1024, 1024], "mean": source_moments[0], "std": source_moments[1]},
        "target_statistics": {"section_count": 11, "pixel_count": 11 * 1024 * 1024,
                              "shape_yx": [1024, 1024], "mean": target_moments[0], "std": target_moments[1],
                              "histogram": reference_histogram.tolist()},
        "scale": scale, "bias": bias, "lut": lut.tolist(),
        "lut_dtype": "uint8",
        "lut_intermediate_levels_dtype": "float32",
        "clip_then_round": True,
        "population_standard_deviation": True,
        "source_histograms_path": "calibration/source_full_histograms.npz",
        "source_inventory_path": "calibration/source_output_inventory.json",
        "source_histograms_sha256": sha256(hist_destination),
        "source_inventory_sha256": sha256(inventory_destination),
        "reference_files": reference_inventory,
        "source_local_z": [0, 2651], "display_local_z": [777, 809],
        "raw_directory": "precomputed/vEMDestripe",
        "final_directory": "precomputed/vEMDestripe_final",
        "checkpoint_sha256": CHECKPOINT_SHA256,
        "checkpoint_reference_block_size": 16,
        "source_context": "Frozen complete 2651-section network output; calibrate with one whole-volume map, then crop the 32 displayed sections",
        "live_demo_context_note": "A new 32-section-only inference must fit its own complete 32-section output; never silently reuse this frozen full-volume source fit",
        "baselines_policy": "Original, EMISAC, FECNet and SSCOR retain their delivered pixels; no evaluator-added calibration",
    }
    record_path = calibration_folder / "vemdestripe_full_volume.json"
    write_json(record_path, record)
    final_folder = output / "precomputed/vEMDestripe_final"
    final_folder.mkdir()
    for z in range(777, 809):
        name = f"slice_{z:05d}.png"
        raw = read_u8(output / "precomputed/vEMDestripe" / name)
        final = lut[raw]
        target = final_folder / name
        Image.fromarray(final).save(target, optimize=True)
        if not np.array_equal(read_u8(target), final):
            raise ValueError(f"Final saved PNG verification failed: Z={z}")
    for name, expected_digest in base_manifest["files"].items():
        if sha256(output / name) != expected_digest:
            raise ValueError(f"A base method/provenance file was accidentally modified: {name}")
    manifest = {key: value for key, value in base_manifest.items() if key != "files"}
    manifest.update(schema=4, schema_version=3, bundle_name=BUNDLE_NAME,
                    default_display="complete_pipeline_outputs",
                    default_metric_protocol="complete_pipeline_without_evaluator_added_normalization",
                    calibrated_methods=["vEMDestripe"],
                    calibration_record="calibration/vemdestripe_full_volume.json",
                    final_output_directories={"Original": "input", "EMISAC": "precomputed/EMISAC",
                                              "FECNet": "precomputed/FECNet", "SSCOR": "precomputed/SSCOR",
                                              "vEMDestripe": "precomputed/vEMDestripe_final"})
    manifest["methods"]["vEMDestripe"] = (
        "Frozen 16x16 released-checkpoint output plus one whole-volume mean/std map fitted on all 2651 outputs "
        "to the 11 recorded original references; this demo crops calibrated local Z [777,809)"
    )
    manifest["processing_contexts"]["vEMDestripe"] = record["source_context"]
    manifest["files"] = {path.relative_to(output).as_posix(): sha256(path)
                         for path in sorted(output.rglob("*")) if path.is_file() and path.name != "manifest.json"}
    write_json(output / "manifest.json", manifest)
    print("All complete-volume statistics, calibrated PNGs and unchanged baselines verified; writing ZIP", flush=True)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, Path(output.name) / path.relative_to(output))
    digest = sha256(archive)
    print(json.dumps({"bundle": str(output), "archive": str(archive), "sha256": digest,
                      "bytes": archive.stat().st_size, "files": len(manifest["files"]),
                      "source_sections": 2651, "target_reference_sections": 11,
                      "scale": scale, "bias": bias}, indent=2), flush=True)
    return archive, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--comparison-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
