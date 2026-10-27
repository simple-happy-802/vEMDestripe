#!/usr/bin/env python3
"""Build the aligned Bumbarger13 release asset used by the Colab demo.

This author-side utility extracts a fixed subset from already completed FECNet,
SSCOR, and vEMDestripe runs. EMISAC is intentionally not packaged: the public
notebook runs the authors' pinned official source. Source paths are explicit
arguments so no private workstation paths are stored in the public project.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import zipfile
from pathlib import Path

import numpy as np
import tifffile
from PIL import Image


METHODS = ("Original", "EMISAC", "FECNet", "SSCOR", "vEMDestripe")


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def to_uint8(image: np.ndarray) -> np.ndarray:
    source = np.asarray(image)
    if source.dtype == np.uint8:
        return source
    maximum = float(np.nanmax(source))
    if maximum <= 255.0:
        return np.clip(np.rint(source), 0, 255).astype(np.uint8)
    if np.issubdtype(source.dtype, np.integer):
        maximum_type = float(np.iinfo(source.dtype).max)
        return np.clip(source.astype(np.float32) / maximum_type * 255.0, 0, 255).astype(np.uint8)
    return np.clip(np.rint(source), 0, 255).astype(np.uint8)


def save_png(image: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(to_uint8(image), mode="L").save(path, optimize=True)


def select_metric_rows(source: Path, destination: Path) -> None:
    allowed = set(METHODS)
    with source.open("r", encoding="utf-8-sig", newline="") as handle:
        rows = [row for row in csv.DictReader(handle) if row.get("method") in allowed]
    present = {row["method"] for row in rows}
    if present != allowed:
        raise ValueError(f"Paper metric table is missing: {sorted(allowed - present)}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    with destination.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument(
        "--fecnet-tif", type=Path, nargs="+", required=True,
        help="Ordered TIFF parts covering the complete Z stack",
    )
    parser.add_argument("--sscor-dir", type=Path, required=True)
    parser.add_argument("--vemdestripe-dir", type=Path, required=True)
    parser.add_argument("--paper-metrics", type=Path, required=True)
    parser.add_argument("--niqe-model", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--z-start", type=int, default=777)
    parser.add_argument("--z-count", type=int, default=32)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output = args.output_dir.resolve()
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f"Refusing to overwrite non-empty output directory: {output}")
    output.mkdir(parents=True, exist_ok=True)
    stop = args.z_start + args.z_count

    for z_index in range(args.z_start, stop):
        name = f"slice_{z_index:05d}.png"
        source = args.input_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "input" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    reference_paths = sorted(args.reference_dir.glob("*.png"))
    if not reference_paths:
        raise FileNotFoundError(f"No reference PNGs in {args.reference_dir}")
    for source in reference_paths:
        destination = output / "reference" / source.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    fec_files = [tifffile.TiffFile(path) for path in args.fecnet_tif]
    try:
        offsets, total = [], 0
        for file in fec_files:
            offsets.append(total)
            total += len(file.pages)
        if stop > total:
            raise IndexError("Requested demo window exceeds the FECNet result")
        for z_index in range(args.z_start, stop):
            for file, offset in zip(fec_files, offsets):
                if z_index < offset + len(file.pages):
                    save_png(
                        file.pages[z_index - offset].asarray(),
                        output / "precomputed" / "FECNet" / f"slice_{z_index:05d}.png",
                    )
                    break
    finally:
        for file in fec_files:
            file.close()

    for z_index in range(args.z_start, stop):
        name = f"slice_{z_index:05d}.png"
        source = args.sscor_dir / name
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "precomputed" / "SSCOR" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

        source = args.vemdestripe_dir / f"slice_{z_index:05d}_deflickered.png"
        if not source.is_file():
            raise FileNotFoundError(source)
        destination = output / "precomputed" / "vEMDestripe" / name
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, destination)

    select_metric_rows(args.paper_metrics, output / "paper_metrics_200_sections.csv")
    shutil.copy2(args.niqe_model, output / "bumbarger13_niqe_model.npz")
    attribution = (
        "Bumbarger13 source: s3://open-neurodata/bumbarger/bumbarger13/image\n"
        "Data license: ODC-By v1.0\n"
        "Dataset page: https://neurodata.io/data/bumbarger13/\n"
        "Cite Bumbarger et al., Cell (2013), and Open NeuroData.\n"
    )
    (output / "DATA_LICENSE.txt").write_text(attribution, encoding="utf-8")

    files = sorted(path for path in output.rglob("*") if path.is_file())
    manifest = {
        "schema": 2,
        "dataset": "Bumbarger13",
        "local_dataset_id": "bumbarger138092_8092_4096_4096tile_r03_c03_1024_stack",
        "public_source": "s3://open-neurodata/bumbarger/bumbarger13/image",
        "data_license": "ODC-By v1.0",
        "local_xy_mip0": {"x": [11164, 12188], "y": [11164, 12188]},
        "demo_local_z": [args.z_start, stop],
        "demo_public_z": [args.z_start + 110, stop + 110],
        "paper_metric_local_z": [777, 977],
        "paper_metric_public_z": [887, 1087],
        "z_display_scale_for_xz_yz": 6,
        "paired_clean_ground_truth": False,
        "methods": {
            "Original": "Aligned input sections",
            "EMISAC": (
                "Not bundled; generated live from official commit "
                "7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5"
            ),
            "FECNet": "Frozen output from the paper-comparison grayscale adaptation",
            "SSCOR": "Frozen output from the dataset-specific seed-42 run",
            "vEMDestripe": "Frozen seed-42 paper-checkpoint output (16x16 reference block)",
        },
        "files": {
            str(path.relative_to(output)).replace("\\", "/"): sha256(path)
            for path in files
        },
    }
    (output / "manifest.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    archive = output.parent / f"{output.name}.zip"
    with zipfile.ZipFile(
        archive, "w", compression=zipfile.ZIP_DEFLATED, compresslevel=9
    ) as zipped:
        for path in sorted(item for item in output.rglob("*") if item.is_file()):
            zipped.write(path, Path(output.name) / path.relative_to(output))
    print(f"Demo directory: {output}")
    print(f"Release archive: {archive}")
    print(f"Archive SHA256: {sha256(archive)}")


if __name__ == "__main__":
    main()
