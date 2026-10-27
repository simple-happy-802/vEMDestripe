"""Verified assets and explicitly labelled evaluation for the official demo.

EMISAC is a precomputed result from unmodified, pinned Python-2 source in a
recorded legacy environment. It is never replaced by the Python-3 port here.
Raw outputs are the default. Bumbarger13 has no paired clean ground truth;
the diagnostics below are not PSNR/SSIM or a biological correctness test.
"""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import stat
import zipfile
from pathlib import Path, PurePosixPath
from typing import Mapping

import numpy as np
import pandas as pd

try:
    from . import demo_metrics as metrics
except ImportError:  # The Colab notebook adds the colab/ folder to sys.path.
    import demo_metrics as metrics


BUNDLE_NAME = "vemdestripe_bumbarger_official_demo_v2"
OFFICIAL_COMMIT = "7a82a17b7bbb4ca14dff2d1c1a4a018fc8cebaf5"
EXPECTED_IDS = tuple(range(777, 809))
EXPECTED_SHAPE = (32, 1024, 1024)  # Z, Y, X
PROVENANCE = "provenance/emisac_official"
OFFICIAL_SOURCE_SHA256 = {
    "EM_normalization.py": "b4ce244ce160c5a2af2a02eb32a1f35f3686cbcccb1e94c70bf80c0ee156865c",
    "normalize_partition.py": "1baa5d4bbc9e6b9aa65a04e8e7bf4b73a17e9d17a3d04b9bc83c33b26e3e090c",
}
_SHA256 = re.compile(r"[0-9a-fA-F]{64}\Z")
MAX_UNCOMPRESSED_BYTES = 4 * 1024**3


class BundleValidationError(ValueError):
    """A downloaded bundle is unsafe, incomplete, or not the recorded demo."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _relative_path(name: str) -> PurePosixPath:
    # Reject Windows drive paths as well as POSIX traversal, regardless of the
    # OS performing extraction. Never reinterpret a backslash as a safe slash.
    if not isinstance(name, str) or not name or "\\" in name or ":" in name:
        raise BundleValidationError(f"Unsafe bundle path: {name!r}")
    path = PurePosixPath(name)
    if path.is_absolute() or any(part in ("", ".", "..") for part in name.split("/")):
        raise BundleValidationError(f"Unsafe bundle path: {name!r}")
    return path


def official_metadata(asset_root: str | Path) -> dict:
    """Read provenance and verify the pinned original source, not a port."""
    root = Path(asset_root)
    record = json.loads((root / PROVENANCE / "metadata.json").read_text(encoding="utf-8"))
    if record.get("original_source_unmodified") is not True:
        raise BundleValidationError("EMISAC must be marked as unmodified original source.")
    if record.get("official_commit") != OFFICIAL_COMMIT:
        raise BundleValidationError("Unexpected official EMISAC commit.")
    if record.get("input_local_z") != [777, 809] or record.get("shape_zyx") != list(EXPECTED_SHAPE):
        raise BundleValidationError("EMISAC provenance does not describe this 32-section window.")
    for filename, expected in OFFICIAL_SOURCE_SHA256.items():
        source = root / PROVENANCE / "source" / filename
        if source.is_symlink() or not source.is_file() or sha256_file(source) != expected:
            raise BundleValidationError(f"Original EMISAC source mismatch: {filename}")
    return record


def verify_bundle(asset_root: str | Path) -> dict:
    """Verify every asset hash and reject unlisted files or symbolic links.

    The archive hash pinned by the notebook is the trust anchor. A manifest
    alone would not authenticate a downloaded archive that was substituted.
    """
    root = Path(asset_root)
    if root.is_symlink() or not root.is_dir():
        raise BundleValidationError(f"Not a real bundle directory: {root}")
    manifest_path = root / "manifest.json"
    if manifest_path.is_symlink():
        raise BundleValidationError("The manifest cannot be a symbolic link.")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("schema_version") != 2 or manifest.get("bundle_name") != BUNDLE_NAME:
        raise BundleValidationError("This is not the official-result v2 demonstration bundle.")
    if manifest.get("input_local_z") != [777, 809] or manifest.get("shape_zyx") != list(EXPECTED_SHAPE):
        raise BundleValidationError("Unexpected bundle geometry or section range.")
    files = manifest.get("files")
    if not isinstance(files, dict) or not files or "manifest.json" in files:
        raise BundleValidationError("Manifest files must map asset paths to SHA256 hashes.")
    required = {
        f"{PROVENANCE}/metadata.json", f"{PROVENANCE}/actual_run.json",
        f"{PROVENANCE}/original.log", f"{PROVENANCE}/normalized_data_final.h5",
        f"{PROVENANCE}/input_stack.tif", "bumbarger13_niqe_model.npz",
        *[f"{PROVENANCE}/source/{name}" for name in OFFICIAL_SOURCE_SHA256],
    }
    if not required.issubset(files):
        raise BundleValidationError(f"Missing provenance assets: {sorted(required - files.keys())}")
    root_resolved = root.resolve()
    for name, expected in files.items():
        relative = _relative_path(name)
        if not isinstance(expected, str) or _SHA256.fullmatch(expected) is None:
            raise BundleValidationError(f"Invalid SHA256 in manifest: {name}")
        path = root.joinpath(*relative.parts)
        if path.is_symlink() or not path.resolve().is_relative_to(root_resolved) or not path.is_file():
            raise BundleValidationError(f"Missing or unsafe asset: {name}")
        if sha256_file(path) != expected.lower():
            raise BundleValidationError(f"Asset checksum mismatch: {name}")
    actual_files = set()
    for path in root.rglob("*"):
        if path.is_symlink():
            raise BundleValidationError(f"Symbolic link in bundle: {path}")
        if path.is_file():
            actual_files.add(path.relative_to(root).as_posix())
    extra = actual_files - set(files) - {"manifest.json"}
    if extra:
        raise BundleValidationError(f"Unlisted files in bundle: {sorted(extra)}")
    official_metadata(root)
    return manifest


def safe_extract_bundle(zip_path: str | Path, destination: str | Path,
                        expected_sha256: str) -> Path:
    """Check a release ZIP, extract without overwrites, and verify its files.

    ``destination`` is the parent of the single top-level bundle directory.
    Existing complete bundles are verified and reused; partial ones are not
    silently repaired or overlaid. The ZIP cannot create files outside that
    directory, and all entries are checked before the first file is extracted.
    """
    archive_path, parent = Path(zip_path), Path(destination)
    if not isinstance(expected_sha256, str) or _SHA256.fullmatch(expected_sha256) is None:
        raise BundleValidationError("A pinned 64-digit release SHA256 is required.")
    if sha256_file(archive_path) != expected_sha256.lower():
        raise BundleValidationError("Release ZIP checksum mismatch; do not use this download.")
    with zipfile.ZipFile(archive_path) as archive:
        total, seen, entries = 0, set(), []
        for info in archive.infolist():
            name = info.filename.rstrip("/") if info.is_dir() else info.filename
            path = _relative_path(name)
            if path.parts[0] != BUNDLE_NAME or (len(path.parts) == 1 and not info.is_dir()):
                raise BundleValidationError("ZIP must contain exactly one named bundle directory.")
            if name in seen:
                raise BundleValidationError(f"Duplicate ZIP entry: {name}")
            seen.add(name)
            file_type = stat.S_IFMT(info.external_attr >> 16)
            if file_type not in (0, stat.S_IFREG, stat.S_IFDIR):
                raise BundleValidationError(f"ZIP symbolic link or special file: {name}")
            total += info.file_size
            if total > MAX_UNCOMPRESSED_BYTES:
                raise BundleValidationError("ZIP exceeds the permitted extracted size.")
            entries.append((info, path))
        if not entries:
            raise BundleValidationError("The ZIP is empty.")
        parent.mkdir(parents=True, exist_ok=True)
        root = parent / BUNDLE_NAME
        if root.exists() or root.is_symlink():
            verify_bundle(root)
            return root
        root.mkdir()  # Exclusive creation: no existing user directory is overlaid.
        try:
            for info, relative in entries:
                target = parent.joinpath(*relative.parts)
                if info.is_dir():
                    target.mkdir(parents=True, exist_ok=True)
                else:
                    target.parent.mkdir(parents=True, exist_ok=True)
                    with archive.open(info) as source, target.open("xb") as output:
                        shutil.copyfileobj(source, output, length=1024 * 1024)
            verify_bundle(root)
        except BaseException:
            # Only this call's newly created, exact bundle directory is removed.
            # Pre-existing directories are handled above and are never deleted.
            shutil.rmtree(root)
            raise
    return root


def load_official_demo(asset_root: str | Path, live_vem_dir: str | Path | None = None,
                       *, verify: bool = True) -> tuple[dict[str, np.ndarray], list[int]]:
    """Load fixed raw PNGs, with an optional explicitly selected live vEM result."""
    root = Path(asset_root)
    if verify:
        verify_bundle(root)
    else:
        official_metadata(root)
    folders = {
        "Original": root / "input",
        "EMISAC": root / "precomputed" / "EMISAC",
        "FECNet": root / "precomputed" / "FECNet",
        "SSCOR": root / "precomputed" / "SSCOR",
        "vEMDestripe": Path(live_vem_dir) if live_vem_dir is not None else root / "precomputed" / "vEMDestripe",
    }
    volumes = {}
    for method in metrics.METHOD_ORDER:
        volume, ids = metrics.load_png_volume(folders[method])
        if tuple(ids) != EXPECTED_IDS or volume.shape != EXPECTED_SHAPE:
            raise BundleValidationError(
                f"{method}: expected local Z 777–808 and ZYX {EXPECTED_SHAPE}; "
                f"got IDs {ids[:2]}…{ids[-2:]} and shape {volume.shape}."
            )
        volumes[method] = volume
    return volumes, list(EXPECTED_IDS)


def verify_emisac_pixels(asset_root: str | Path) -> dict:
    """Independently compare displayed PNG values to the original float H5.

    This checks the declared conversion, not equality with the author's
    historical machine, and never performs photometric matching.
    """
    import h5py

    root = Path(asset_root)
    record = official_metadata(root)
    h5_path = root / PROVENANCE / "normalized_data_final.h5"
    if record.get("raw_h5_sha256") != sha256_file(h5_path):
        raise BundleValidationError("Original float H5 differs from its run provenance.")
    with h5py.File(h5_path, "r") as handle:
        raw = np.asarray(handle["stack"][()])
    expected_hwz = (*EXPECTED_SHAPE[1:], EXPECTED_SHAPE[0])
    if raw.shape != expected_hwz or not np.isfinite(raw).all():
        raise BundleValidationError("Invalid original H5 geometry or nonfinite values.")
    png, ids = metrics.load_png_volume(root / "precomputed" / "EMISAC")
    expected = np.clip(np.rint(raw), 0, 255).astype(np.uint8).transpose(2, 0, 1)
    if tuple(ids) != EXPECTED_IDS or not np.array_equal(png, expected):
        raise BundleValidationError("Displayed EMISAC PNGs are not the declared raw-H5 conversion.")
    return {"png_values_identical_to_raw_h5_quantization": True,
            "h5_axis_order": "HWZ", "h5_shape": list(raw.shape),
            "float_range": [float(raw.min()), float(raw.max())],
            "png_quantization": "clip(rint(H5),0,255).astype(uint8)",
            "additional_photometric_matching": False}


def provenance_table(asset_root: str | Path, live_vem_dir: str | Path | None = None) -> pd.DataFrame:
    """Return visible result origins; frozen adaptations are not called original."""
    root = Path(asset_root)
    manifest = json.loads((root / "manifest.json").read_text(encoding="utf-8"))
    metadata = official_metadata(root)
    descriptions = dict(manifest.get("methods", {}))
    descriptions["EMISAC"] = "Precomputed: unmodified official Python-2 source + recorded legacy environment"
    if live_vem_dir is not None:
        descriptions["vEMDestripe"] = "Live inference with the released vEMDestripe checkpoint"
    return pd.DataFrame([
        {"method": name, "result_source": descriptions.get(name, "See bundle provenance"),
         "display_local_z": "777–808 (32 sections)",
         "official_commit": metadata["official_commit"] if name == "EMISAC" else ""}
        for name in metrics.METHOD_ORDER
    ]).set_index("method")


def evaluate_outputs(volumes: Mapping[str, np.ndarray], niqe_model: str | Path,
                     *, mode: str = "raw", xy_samples: int = 8,
                     cross_sections: int = 6, z_ratio: float = 6.0,
                     border: int = 32) -> pd.DataFrame:
    """Evaluate raw or explicitly global-matched outputs without hidden matching.

    Native-Z diagnostics are always evaluated before cross-plane interpolation.
    NIQE XZ/YZ uses the stated Z enlargement only to obtain supported planes;
    it must not be confused with native-Z continuity. Lower or higher values
    alone do not establish preservation of biological ultrastructure.
    """
    if mode not in ("raw", "global_affine_matched"):
        raise ValueError("mode must be 'raw' or 'global_affine_matched'")
    if xy_samples <= 0 or cross_sections <= 0 or z_ratio <= 0 or border < 0:
        raise ValueError("Invalid evaluation sampling settings.")
    for name in metrics.METHOD_ORDER:
        volume = volumes[name]
        if volume.ndim != 3 or len(volume) < 2 or min(volume.shape[1:]) <= 2 * border:
            raise ValueError(f"Invalid volume geometry for {name}: {volume.shape}")
    if mode == "raw":
        selected = volumes
        transforms = {name: (1.0, 0.0) for name in metrics.METHOD_ORDER}
    else:
        selected, transforms = metrics.matched_volumes(volumes)
    with np.load(niqe_model) as model:
        mu, cov = model["mu"], model["cov"]
    rows = []
    for method in metrics.METHOD_ORDER:
        volume = selected[method]
        indices = metrics._spaced(len(volume), xy_samples)
        scores = [metrics.niqe_score(volume[index], mu, cov, 96) for index in indices]
        finite = [score for score in scores if np.isfinite(score)]
        profile = volume.mean(axis=(1, 2), dtype=np.float64)
        rows.append({
            "method": method, "output_mode": mode, "demo_sections": len(volume),
            "ASID_MAD": metrics.asid_z_mean_mad_diff(profile),
            "local_ASID": metrics.local_asid(volume),
            "Z_HF_percent": metrics.z_brightness_hf_percent(profile),
            "NIQE_XY": float(np.mean(finite)) if finite else float("nan"),
            "NIQE_XZ": metrics._niqe_orientation(volume, "xz", cross_sections, mu, cov, 96, z_ratio, border),
            "NIQE_YZ": metrics._niqe_orientation(volume, "yz", cross_sections, mu, cov, 96, z_ratio, border),
            "contrast_index": float(np.mean([metrics.contrast_index(volume[i]) for i in indices])),
            "blur_index": float(np.mean([metrics.blur_index(volume[i]) for i in indices])),
            "global_match_scale": transforms[method][0],
            "global_match_bias": transforms[method][1],
            "xy_samples": len(indices), "cross_sections": cross_sections,
            "cross_plane_niqe_z_ratio": z_ratio,
        })
    return pd.DataFrame(rows).set_index("method").loc[list(metrics.METHOD_ORDER)]


def evaluate_both(volumes: Mapping[str, np.ndarray], niqe_model: str | Path,
                   **kwargs) -> pd.DataFrame:
    """Return two separately labelled evaluations, never a mixed-mode ranking."""
    return pd.concat([
        evaluate_outputs(volumes, niqe_model, mode=mode, **kwargs)
        for mode in ("raw", "global_affine_matched")
    ])
