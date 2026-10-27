"""Paper-aligned metrics and visualisation helpers for the Colab demo.

Bumbarger13 is real data without slice-wise clean targets. This module
therefore does not report PSNR or SSIM. Values computed here use a fixed
32-section demonstration window and must not replace the paper's 200-section
results.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from PIL import Image
from scipy.ndimage import gaussian_filter1d, zoom

from evaluation.niqe import niqe_score


METHOD_ORDER = ("Original", "EMISAC", "FECNet", "SSCOR", "vEMDestripe")
COLORS = {
    "Original": "#7F7F7F",
    "EMISAC": "#0072B2",
    "FECNet": "#E69F00",
    "SSCOR": "#5E3C99",
    "vEMDestripe": "#D55E00",
}


def _section_id(path: Path) -> int:
    match = re.search(r"slice_(\d+)", path.stem)
    if not match:
        raise ValueError(f"Cannot parse a section index from {path.name}")
    return int(match.group(1))


def load_png_volume(folder: str | Path) -> tuple[np.ndarray, list[int]]:
    """Load an aligned uint8 ZYX volume from a PNG directory."""
    folder = Path(folder)
    paths = sorted(folder.glob("slice_*.png"), key=_section_id)
    if not paths:
        raise FileNotFoundError(f"No slice_*.png files found in {folder}")
    indices = [_section_id(path) for path in paths]
    if len(indices) != len(set(indices)):
        raise ValueError(f"Duplicate section indices in {folder}")
    volume = np.stack([
        np.asarray(Image.open(path).convert("L"), dtype=np.uint8) for path in paths
    ])
    return volume, indices


def load_demo_volumes(asset_root: str | Path, live_emisac_dir: str | Path,
                      live_vem_dir: str | Path | None = None
                      ) -> tuple[dict[str, np.ndarray], list[int]]:
    """Load aligned volumes, requiring official live EMISAC output.

    FECNet and SSCOR are frozen paper-comparison outputs. vEMDestripe prefers a
    live result when supplied and otherwise uses the packaged checkpoint
    fallback. EMISAC has no packaged fallback: ``live_emisac_dir`` must contain
    the PNGs emitted by the official-source compatibility runner.
    """
    root = Path(asset_root)
    emisac_folder = Path(live_emisac_dir)
    if not emisac_folder.is_dir():
        raise FileNotFoundError(
            "Official EMISAC output directory is missing: "
            f"{emisac_folder}. Run the official-source EMISAC cell first."
        )
    folders = {
        "Original": root / "input",
        "EMISAC": emisac_folder,
        "FECNet": root / "precomputed" / "FECNet",
        "SSCOR": root / "precomputed" / "SSCOR",
        "vEMDestripe": (
            Path(live_vem_dir) if live_vem_dir else root / "precomputed" / "vEMDestripe"
        ),
    }
    volumes: dict[str, np.ndarray] = {}
    reference_ids: list[int] | None = None
    reference_shape: tuple[int, ...] | None = None
    for method in METHOD_ORDER:
        volume, indices = load_png_volume(folders[method])
        if reference_ids is None:
            reference_ids = indices
            reference_shape = volume.shape
        elif indices != reference_ids:
            raise ValueError(f"{method} indices do not match Original")
        elif volume.shape != reference_shape:
            raise ValueError(
                f"{method} shape {volume.shape} does not match Original "
                f"shape {reference_shape}; aligned ZYX volumes are required"
            )
        volumes[method] = volume
    assert reference_ids is not None
    return volumes, reference_ids


def global_affine_match(source: np.ndarray, target: np.ndarray
                        ) -> tuple[np.ndarray, float, float]:
    """Match one complete demo volume to Original with one scale and bias."""
    source_f = source.astype(np.float64)
    target_f = target.astype(np.float64)
    source_std = float(source_f.std())
    if source_std < 1e-8:
        raise ValueError("Cannot photometrically match a zero-contrast volume")
    scale = float(target_f.std()) / source_std
    bias = float(target_f.mean()) - scale * float(source_f.mean())
    matched = np.clip(np.rint(source_f * scale + bias), 0, 255).astype(np.uint8)
    return matched, scale, bias


def matched_volumes(volumes: Mapping[str, np.ndarray]
                    ) -> tuple[dict[str, np.ndarray], dict[str, tuple[float, float]]]:
    """Apply the paper's one-transform-per-volume photometric protocol."""
    target = volumes["Original"]
    output = {"Original": target}
    transforms = {"Original": (1.0, 0.0)}
    for method in METHOD_ORDER[1:]:
        output[method], scale, bias = global_affine_match(volumes[method], target)
        transforms[method] = (scale, bias)
    return output, transforms


def asid_z_mean_mad_diff(profile: Sequence[float]) -> float:
    """MAD of adjacent native-Z whole-section mean changes."""
    difference = np.diff(np.asarray(profile, dtype=np.float64))
    if not difference.size:
        return 0.0
    return float(np.median(np.abs(difference - np.median(difference))))


def z_brightness_hf_percent(profile: Sequence[float]) -> float:
    profile = np.asarray(profile, dtype=np.float64)
    high = profile - gaussian_filter1d(profile, sigma=3, mode="nearest")
    return float(np.sqrt(np.mean(high * high)) / 255.0 * 100.0)


def local_asid(volume: np.ndarray) -> float:
    """Robust blockwise median jump normalised by local IQR."""
    values: list[float] = []
    block = max(32, min(volume.shape[1:]) // 8)
    for image_a, image_b in zip(volume[:-1], volume[1:]):
        for y in range(0, image_a.shape[0] - block + 1, block):
            for x in range(0, image_a.shape[1] - block + 1, block):
                a = image_a[y:y + block, x:x + block].astype(np.float32)
                b = image_b[y:y + block, x:x + block].astype(np.float32)
                iqr_a = float(np.subtract(*np.percentile(a, [75, 25])))
                iqr_b = float(np.subtract(*np.percentile(b, [75, 25])))
                jump = abs(float(np.median(a)) - float(np.median(b)))
                values.append(jump / max((iqr_a + iqr_b) / 2.0, 1.0))
    return float(np.mean(values))


def contrast_index(image: np.ndarray) -> float:
    rms = float(np.asarray(image, dtype=np.float32).std()) / 255.0
    return float(min(rms / 0.3, 1.0))


def blur_index(image: np.ndarray, kernel_size: int = 9) -> float:
    """Gradient-loss sharpness diagnostic used in the paper."""
    source = np.asarray(image, dtype=np.float32) / 255.0
    blurred = cv2.blur(
        source, (kernel_size, kernel_size), borderType=cv2.BORDER_REFLECT_101
    )
    sx = cv2.Sobel(source, cv2.CV_32F, 1, 0, ksize=3,
                   borderType=cv2.BORDER_REFLECT_101)
    sy = cv2.Sobel(source, cv2.CV_32F, 0, 1, ksize=3,
                   borderType=cv2.BORDER_REFLECT_101)
    bx = cv2.Sobel(blurred, cv2.CV_32F, 1, 0, ksize=3,
                   borderType=cv2.BORDER_REFLECT_101)
    by = cv2.Sobel(blurred, cv2.CV_32F, 0, 1, ksize=3,
                   borderType=cv2.BORDER_REFLECT_101)
    gradient = cv2.magnitude(sx, sy)
    denominator = float(gradient.sum(dtype=np.float64))
    if denominator <= 1e-12:
        return 0.0
    loss = np.maximum(gradient - cv2.magnitude(bx, by), 0.0)
    return float(np.clip(loss.sum(dtype=np.float64) / denominator, 0.0, 1.0))


def _spaced(length: int, count: int) -> list[int]:
    count = min(length, count)
    return sorted(set(np.rint(np.linspace(0, length - 1, count)).astype(int).tolist()))


def _niqe_orientation(volume: np.ndarray, orientation: str, positions: int,
                      mu: np.ndarray, cov: np.ndarray, patch: int,
                      z_ratio: float, border: int) -> float:
    side = volume.shape[1] if orientation == "xz" else volume.shape[2]
    locations = [border + value for value in _spaced(side - 2 * border, positions)]
    scores = []
    for location in locations:
        section = (
            volume[:, location, :] if orientation == "xz"
            else volume[:, :, location]
        )
        if border:
            section = section[:, border:-border]
        section = zoom(section.astype(np.float32), (z_ratio, 1.0), order=1)
        scores.append(niqe_score(section, mu, cov, patch))
    finite = [score for score in scores if np.isfinite(score)]
    return float(np.mean(finite)) if finite else float("nan")


def evaluate_demo(volumes: Mapping[str, np.ndarray], niqe_model: str | Path,
                  xy_samples: int = 8, cross_sections: int = 6,
                  z_ratio: float = 6.0, border: int = 32) -> pd.DataFrame:
    """Evaluate the demo after paper-consistent global affine matching."""
    saved = np.load(niqe_model)
    mu, cov, patch = saved["mu"], saved["cov"], 96
    aligned, transforms = matched_volumes(volumes)
    rows = []
    for method in METHOD_ORDER:
        volume = aligned[method]
        xy_ids = _spaced(len(volume), xy_samples)
        xy_scores = [niqe_score(volume[index], mu, cov, patch) for index in xy_ids]
        profile = volume.mean(axis=(1, 2), dtype=np.float64)
        rows.append({
            "method": method,
            "demo_sections": len(volume),
            "ASID_MAD": asid_z_mean_mad_diff(profile),
            "local_ASID": local_asid(volume),
            "Z_HF_percent": z_brightness_hf_percent(profile),
            "NIQE_XY": float(np.nanmean(xy_scores)),
            "NIQE_XZ": _niqe_orientation(
                volume, "xz", cross_sections, mu, cov, patch, z_ratio, border
            ),
            "NIQE_YZ": _niqe_orientation(
                volume, "yz", cross_sections, mu, cov, patch, z_ratio, border
            ),
            "contrast_index": float(np.mean([
                contrast_index(volume[index]) for index in xy_ids
            ])),
            "blur_index": float(np.mean([
                blur_index(volume[index]) for index in xy_ids
            ])),
            "global_match_scale": transforms[method][0],
            "global_match_bias": transforms[method][1],
        })
    return pd.DataFrame(rows).set_index("method").loc[list(METHOD_ORDER)]


def plot_xy(volumes: Mapping[str, np.ndarray], indices: Sequence[int],
            position: int) -> None:
    fig, axes = plt.subplots(
        1, len(METHOD_ORDER), figsize=(19, 4), constrained_layout=True
    )
    for axis, method in zip(axes, METHOD_ORDER):
        axis.imshow(volumes[method][position], cmap="gray", vmin=0, vmax=255)
        axis.set_title(method)
        axis.axis("off")
    fig.suptitle(f"Bumbarger13 — local Z={indices[position]}", fontsize=14)
    plt.show()


def plot_cross_sections(volumes: Mapping[str, np.ndarray], axis: str = "xz",
                        position: int = 512, z_ratio: float = 6.0) -> None:
    fig, axes = plt.subplots(
        1, len(METHOD_ORDER), figsize=(19, 2), constrained_layout=True
    )
    for panel, method in zip(axes, METHOD_ORDER):
        volume = volumes[method]
        section = (
            volume[:, position, :] if axis == "xz"
            else volume[:, :, position]
        )
        section = zoom(section.astype(np.float32), (z_ratio, 1.0), order=1)
        # Preserve the explicitly requested Z magnification. ``aspect='auto'``
        # would stretch the thin demo plane again to fill a square subplot.
        panel.imshow(section, cmap="gray", vmin=0, vmax=255, aspect="equal")
        panel.set_title(method)
        panel.axis("off")
    fig.suptitle(
        f"{axis.upper()} section at position {position}; Z displayed at {z_ratio:g}×",
        fontsize=14,
    )
    plt.show()


def plot_z_profiles(volumes: Mapping[str, np.ndarray], indices: Sequence[int]) -> None:
    plt.figure(figsize=(11, 5))
    for method in METHOD_ORDER:
        profile = volumes[method].mean(axis=(1, 2), dtype=np.float64)
        plt.plot(indices, profile, label=method, color=COLORS[method], linewidth=2)
    plt.xlabel("Local section index")
    plt.ylabel("Whole-section mean intensity")
    plt.title("Native-Z brightness profiles (demo window)")
    plt.grid(alpha=0.25)
    plt.legend(ncol=3)
    plt.show()
