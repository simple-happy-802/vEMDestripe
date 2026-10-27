#!/usr/bin/env python3
"""Run unmodified official EMISAC in a fixed Linux container and compare H5."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

COMMIT = "7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5"
SOURCE_SHA256 = {
    "EM_normalization.py": "b4ce244ce160c5a2af2a02eb32a1f35f3686cbcccb1e94c70bf80c0ee156865c",
    "normalize_partition.py": "1baa5d4bbc9e6b9aa65a04e8e7bf4b73a17e9d17a3d04b9bc83c33b26e3e090c",
}


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            value.update(block)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--asset-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--image", default="vemdestripe-emisac-original:demo-v2")
    parser.add_argument("--docker", default="docker")
    parser.add_argument("--build", action="store_true")
    args = parser.parse_args()
    root, destination = args.asset_root.resolve(), args.output_dir.resolve()
    provenance = root / "provenance/emisac_official"
    metadata = json.loads((provenance / "metadata.json").read_text(encoding="utf-8"))
    if not metadata["original_source_unmodified"] or metadata["official_commit"] != COMMIT:
        raise ValueError("Only unmodified original-source assets are accepted")
    for name, expected in SOURCE_SHA256.items():
        if metadata["source_sha256"].get(name) != expected or digest(provenance / "source" / name) != expected:
            raise ValueError(f"Source checksum mismatch: {name}")
    input_path = provenance / "input_stack.tif"
    reference_path = provenance / "normalized_data_final.h5"
    if digest(input_path) != metadata["input_tiff_sha256"]:
        raise ValueError("Input checksum mismatch")
    if digest(reference_path) != metadata["raw_h5_sha256"]:
        raise ValueError("Reference output checksum mismatch")
    if destination.exists():
        raise FileExistsError(f"Refusing to overwrite output directory: {destination}")
    if args.build:
        subprocess.run([args.docker, "build", "--platform", "linux/amd64", "-t", args.image,
                        str(Path(__file__).resolve().parent)], check=True)
    destination.mkdir(parents=True)
    source = destination / "source"
    source.mkdir()
    for name in (*SOURCE_SHA256, "license.txt", "README.md"):
        shutil.copy2(provenance / "source" / name, source / name)
    for name in SOURCE_SHA256:
        (source / name).chmod(0o755)  # Metadata only; source bytes remain unchanged.
    image_id = subprocess.check_output(
        [args.docker, "image", "inspect", args.image, "--format", "{{.Id}}"], text=True
    ).strip()
    output = destination / "output"
    output.mkdir()
    command = [args.docker, "run", "--rm", "--platform", "linux/amd64", "--network", "none",
               "--cpus", "2", "--memory", "8g", "--shm-size", "1g",
               "--mount", f"type=bind,source={source},target=/official,readonly",
               "--mount", f"type=bind,source={input_path},target=/input_stack.tif,readonly",
               "--mount", f"type=bind,source={output},target=/output",
               "--workdir", "/official", "--env", "PYTHONUNBUFFERED=1",
               "--env", "PYTHONHASHSEED=42", args.image,
               "python", "/official/EM_normalization.py", "--up_fg", "0", "--up_z", "1",
               "--fact", "1e10", "--lm", "100", "/input_stack.tif",
               "4", "4", "32", "32", "4", "4", "2", "/output"]
    print("Running original source; progress is written to run.log", flush=True)
    with (destination / "run.log").open("wb") as log:
        subprocess.run(command, check=True, stdout=log, stderr=subprocess.STDOUT)
    with h5py.File(output / "normalized_data_final.h5", "r") as handle:
        current = np.asarray(handle["stack"][:], dtype=np.float64)
    with h5py.File(reference_path, "r") as handle:
        reference = np.asarray(handle["stack"][:], dtype=np.float64)
    if current.shape != reference.shape or not np.isfinite(current).all():
        raise ValueError("Invalid output shape or non-finite values")
    delta = np.abs(current - reference)
    report = {"official_source_unmodified": True, "shape_hwz": list(current.shape),
              "exact_equal": bool(np.array_equal(current, reference)),
              "mean_absolute_error": float(delta.mean()),
              "max_absolute_error": float(delta.max()),
              "rmse": float(np.sqrt(np.mean(delta ** 2))),
              "image": args.image, "docker_image_id": image_id, "command": command}
    (destination / "comparison.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    png = destination / "png"
    png.mkdir()
    for offset, z in enumerate(range(*metadata["input_local_z"])):
        Image.fromarray(np.clip(np.rint(current[:, :, offset]), 0, 255).astype(np.uint8)).save(
            png / f"slice_{z:05d}.png")
    print(json.dumps({key: value for key, value in report.items() if key != "command"}, indent=2))


if __name__ == "__main__":
    main()
