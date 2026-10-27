#!/usr/bin/env python3
"""Package independently checked, unmodified-Python-2 EMISAC demo results.

Author-side utility, not a public inference implementation. The old demo is
copied without altering its input or frozen baseline images. EMISAC PNGs are
generated only by fixed round/clip quantization of the recorded original H5.
The complete raw H5 and run provenance are kept in the release asset.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

COMMIT = "7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5"
SOURCE_SHA256 = {
    "EM_normalization.py": "b4ce244ce160c5a2af2a02eb32a1f35f3686cbcccb1e94c70bf80c0ee156865c",
    "normalize_partition.py": "1baa5d4bbc9e6b9aa65a04e8e7bf4b73a17e9d17a3d04b9bc83c33b26e3e090c",
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def write_json(path: Path, value: dict) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def validate_old_bundle(root: Path) -> dict:
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    for name, expected in manifest["files"].items():
        file_path = root / name
        if not file_path.is_file() or sha256(file_path) != expected:
            raise ValueError(f"Old demo file failed checksum: {name}")
    if manifest["demo_local_z"] != [777, 809]:
        raise ValueError("Expected the audited local Z [777,809) demo")
    return manifest


def build(args: argparse.Namespace) -> tuple[Path, str]:
    old = args.base_bundle.resolve()
    test = args.test_root.resolve()
    output = args.output_dir.resolve()
    archive = output.with_suffix(".zip")
    if output.exists() or archive.exists():
        raise FileExistsError("Refusing to overwrite an existing bundle or archive")
    old_manifest = validate_old_bundle(old)
    original = test / "cases/real32/git_blob_lf_v1"
    run = json.loads((original / "original_run.json").read_text(encoding="utf-8"))
    if run["exit_code"] != 0:
        raise ValueError("The reference run was not successful")
    sources = test / "source_original_git_blob_lf"
    for name, expected in SOURCE_SHA256.items():
        if sha256(sources / name) != expected or run["source_sha256"][name] != expected:
            raise ValueError(f"Unmodified upstream source checksum mismatch: {name}")
    input_tif = original / "input_stack.tif"
    if sha256(input_tif) != run["input_sha256"]:
        raise ValueError("Original input TIFF checksum mismatch")
    h5_path = original / "original/normalized_data_final.h5"
    with h5py.File(h5_path, "r") as handle:
        raw = np.asarray(handle["stack"][:])
    if raw.shape != (1024, 1024, 32) or not np.isfinite(raw).all():
        raise ValueError("Invalid reference output: expected finite H,W,Z=1024,1024,32")
    # Explicitly verify TIFF versus the published PNG input, not only its hash.
    with Image.open(input_tif) as image:
        if image.n_frames != 32:
            raise ValueError("Original reference input must contain 32 frames")
        for offset, z in enumerate(range(777, 809)):
            image.seek(offset)
            with Image.open(old / "input" / f"slice_{z:05d}.png") as expected:
                if not np.array_equal(np.asarray(image.convert("L")), np.asarray(expected.convert("L"))):
                    raise ValueError(f"Input differs from original reference run at local Z {z}")

    output.mkdir(parents=True)
    for name in ("input", "reference"):
        shutil.copytree(old / name, output / name)
    for name in ("FECNet", "SSCOR", "vEMDestripe"):
        shutil.copytree(old / "precomputed" / name, output / "precomputed" / name)
    for name in ("DATA_LICENSE.txt", "bumbarger13_niqe_model.npz", "paper_metrics_200_sections.csv"):
        shutil.copy2(old / name, output / name)
    png_dir = output / "precomputed/EMISAC"
    png_dir.mkdir(parents=True)
    png_values = np.clip(np.rint(raw), 0, 255).astype(np.uint8)
    for offset, z in enumerate(range(777, 809)):
        path = png_dir / f"slice_{z:05d}.png"
        Image.fromarray(png_values[:, :, offset]).save(path, optimize=True)
        with Image.open(path) as image:
            if not np.array_equal(np.asarray(image), png_values[:, :, offset]):
                raise ValueError(f"PNG quantization verification failed at Z {z}")

    provenance = output / "provenance/emisac_official"
    source_output = provenance / "source"
    source_output.mkdir(parents=True)
    for name in (*SOURCE_SHA256, "license.txt", "README.md"):
        shutil.copy2(sources / name, source_output / name)
    shutil.copy2(h5_path, provenance / "normalized_data_final.h5")
    shutil.copy2(input_tif, provenance / "input_stack.tif")
    shutil.copy2(original / "original.log", provenance / "original.log")
    # Preserve the numerical command/environment while omitting private host paths.
    public_run = dict(run)
    command = public_run["command"]
    main_index = command.index("/official/EM_normalization.py")
    public_run["command"] = ["python", *command[main_index:]]
    public_run["command_note"] = (
        "Numerical command preserved; host Docker executable, private bind paths, "
        "and container name omitted. Both source and input were read-only mounts."
    )
    public_run["resource_limits"] = {"cpus": 2, "memory_gib": 8, "shm_gib": 1}
    write_json(provenance / "actual_run.json", public_run)
    metadata = {
        "schema_version": 2,
        "method": "EMISAC",
        "implementation": "official_python2_original",
        "display_label": "Official original source, fixed legacy environment (precomputed)",
        "original_source_unmodified": True,
        "official_repository": "https://github.com/azadis/EMISAC",
        "official_commit": COMMIT,
        "input_local_z": [777, 809],
        "shape_zyx": [32, 1024, 1024],
        "h5_axis_order": "HWZ",
        "processing_context": "Only the published 32-section input window was processed",
        "parameters": {
            "step_xy": [4, 4], "initial_window_xy": [32, 32],
            "partition_xy": [4, 4], "workers": 2,
            "lm": 100, "fact": 1e10, "up_fg": 0, "up_z": 1, "cns_fg": 0,
        },
        "all_upstream_defaults": False,
        "parameter_note": "Z upsampling disabled explicitly; up_z inactive; no contrast enhancement",
        "environment": run["environment"],
        "reference_container_image_id": run["docker_image_id"],
        "source_sha256": SOURCE_SHA256,
        "input_tiff_sha256": sha256(input_tif),
        "raw_h5_sha256": sha256(h5_path),
        "raw_h5_relative_path": "provenance/emisac_official/normalized_data_final.h5",
        "input_tiff_relative_path": "provenance/emisac_official/input_stack.tif",
        "png_quantization": "clip(rint(H5),0,255).astype(uint8)",
        "png_pixel_values_verified_against_raw_h5": True,
        "postprocessing": "No additional affine matching, histogram matching, or per-slice scaling",
        "environment_note": (
            "Runnable documented historical dependency environment, not a claim of "
            "recovering the authors' complete historical machine. Rebuilt-container "
            "outputs must be compared before claiming numerical identity."
        ),
        "python3_port_policy": "Not used as the official reference output in this bundle",
    }
    write_json(provenance / "metadata.json", metadata)
    # Existing historical table remains an explicitly separate provenance artifact.
    manifest = {key: value for key, value in old_manifest.items() if key != "files"}
    manifest.update({
        "schema": 3, "schema_version": 2, "bundle_name": output.name,
        "input_local_z": [777, 809], "shape_zyx": [32, 1024, 1024],
        "default_display": "raw", "default_metric_protocol": "raw",
        "historical_paper_table_only": True,
        "historical_paper_table_note": (
            "paper_metrics_200_sections.csv is an inherited historical table, not "
            "recomputed here and not the official 32-section demonstration result."
        ),
        "original_bundle_input_hashes_preserved": True,
    })
    manifest["methods"]["EMISAC"] = (
        "Frozen result from unmodified official Python 2 source, fixed legacy "
        "environment, local Z [777,809); not the Python 3 compatibility port"
    )
    manifest["processing_contexts"] = {
        "EMISAC": "Optimization on exactly the 32-section demo window",
        "FECNet": "Inherited paper-comparison grayscale adaptation; extracted result window",
        "SSCOR": "Inherited dataset-specific seed-42 result; extracted result window",
        "vEMDestripe": "Inherited paper-checkpoint output; optional live inference is separately labelled",
    }
    manifest["files"] = {
        path.relative_to(output).as_posix(): sha256(path)
        for path in sorted(output.rglob("*")) if path.is_file()
    }
    write_json(output / "manifest.json", manifest)
    with zipfile.ZipFile(archive, "w", zipfile.ZIP_DEFLATED, compresslevel=6) as zipped:
        for path in sorted(output.rglob("*")):
            if path.is_file():
                zipped.write(path, Path(output.name) / path.relative_to(output))
    digest = sha256(archive)
    print(json.dumps({"bundle": str(output), "archive": str(archive), "sha256": digest,
                      "files": len(manifest["files"]), "bytes": archive.stat().st_size}, indent=2))
    return archive, digest


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-bundle", type=Path, required=True)
    parser.add_argument("--test-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    build(parser.parse_args())


if __name__ == "__main__":
    main()
