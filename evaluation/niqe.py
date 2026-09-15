#!/usr/bin/env python3
"""Domain-calibrated NIQE for EM image folders.

This implements the original two-scale 36-D NIQE feature construction and is
intended for *within-domain* raw-versus-processed comparisons.  A set of
unpaired normal EM images is used only to fit the pristine MVG model; they are
not treated as paired ground truth.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from scipy import linalg, ndimage, special

try:
    import torch
    import torch.nn.functional as torch_f
except ImportError:  # CPU-only environments retain the reference implementation.
    torch = None
    torch_f = None


Array = np.ndarray
NIQE_IMPLEMENTATION_ID = "live_official_circshift_twoscale_v2"
GAUSS = cv2.getGaussianKernel(7, 7 / 6) @ cv2.getGaussianKernel(7, 7 / 6).T
GAUSS = (GAUSS / GAUSS.sum()).astype(np.float32)
GAMMA_GRID = np.arange(0.2, 10.001, 0.001)
R_GAMMA = special.gamma(2 / GAMMA_GRID) ** 2 / (
    special.gamma(1 / GAMMA_GRID) * special.gamma(3 / GAMMA_GRID)
)
AGGD_SCALE_GRID = np.sqrt(special.gamma(1 / GAMMA_GRID) / special.gamma(3 / GAMMA_GRID))


def read_gray(path: Path) -> Array:
    """Unicode-safe OpenCV grayscale reader for Windows paths."""
    raw = np.fromfile(os.fspath(path), dtype=np.uint8)
    image = cv2.imdecode(raw, cv2.IMREAD_GRAYSCALE)
    if image is None:
        raise ValueError(f"Cannot read image: {path}")
    return image


def slice_id(path: Path) -> int:
    match = re.search(r"slice_(\d+)", path.name)
    if match is None:
        raise ValueError(f"Expected a slice_XXXXX filename: {path.name}")
    return int(match.group(1))


def indexed_paths(folder: Path, pattern: str) -> Dict[int, Path]:
    paths: Dict[int, Path] = {}
    for path in folder.glob(pattern):
        paths[slice_id(path)] = path
    if not paths:
        raise ValueError(f"No files matching {pattern} in {folder}")
    return paths


def evenly_spaced(values: Sequence[int], count: int) -> List[int]:
    if count <= 0:
        raise ValueError("count must be positive")
    if len(values) <= count:
        return list(values)
    idx = np.linspace(0, len(values) - 1, count).round().astype(int)
    return [values[i] for i in idx]


def mscn_and_sigma(image: Array) -> Tuple[Array, Array]:
    image = image.astype(np.float32)
    mu = ndimage.convolve(image, GAUSS, mode="nearest")
    sigma = np.sqrt(np.abs(ndimage.convolve(image * image, GAUSS, mode="nearest") - mu * mu))
    return (image - mu) / (sigma + 1.0), sigma


def estimate_aggd(vector: Array) -> Tuple[float, float, float]:
    x = np.asarray(vector, dtype=np.float64).ravel()
    x = x[np.isfinite(x)]
    if x.size < 10:
        return 1.0, 0.0, 0.0
    left, right = x[x < 0], x[x > 0]
    left_std = np.sqrt(np.mean(left * left)) if left.size else 0.0
    right_std = np.sqrt(np.mean(right * right)) if right.size else 0.0
    if left_std == 0.0 and right_std == 0.0:
        return 1.0, 0.0, 0.0
    gammahat = left_std / (right_std + 1e-12)
    rhat = np.mean(np.abs(x)) ** 2 / (np.mean(x * x) + 1e-12)
    rhatnorm = rhat * ((gammahat ** 3 + 1) * (gammahat + 1)) / (gammahat ** 2 + 1) ** 2
    # R_GAMMA is monotonic increasing on the predefined gamma grid.  The
    # former implementation scanned all 9,801 candidates for every AGGD fit;
    # binary search plus an exact two-neighbour comparison selects the *same*
    # discrete grid value in O(log N).  This matters when Fig. 5-style
    # histograms score hundreds of XZ/YZ cross-sections.
    right_index = int(np.searchsorted(R_GAMMA, rhatnorm, side="left"))
    if right_index <= 0:
        grid_index = 0
    elif right_index >= len(R_GAMMA):
        grid_index = len(R_GAMMA) - 1
    else:
        left_index = right_index - 1
        grid_index = left_index if abs(R_GAMMA[left_index] - rhatnorm) <= abs(R_GAMMA[right_index] - rhatnorm) else right_index
    alpha = float(GAMMA_GRID[grid_index])
    scale = np.sqrt(special.gamma(1 / alpha) / special.gamma(3 / alpha))
    return alpha, left_std * scale, right_std * scale


def feature_18(mscn_patch: Array) -> Array:
    alpha, beta_l, beta_r = estimate_aggd(mscn_patch)
    feat: List[float] = [alpha, (beta_l + beta_r) / 2.0]
    # Match the authors' public LIVE MATLAB release exactly: computefeature.m
    # constructs all four paired-product fields with ``circshift``.  Although
    # a valid-neighbour implementation is also defensible, it defines a
    # different feature distribution and must not be mixed with LIVE-trained
    # or LIVE-style pristine models.
    products = (
        mscn_patch * np.roll(mscn_patch, 1, axis=1),
        mscn_patch * np.roll(mscn_patch, 1, axis=0),
        mscn_patch * np.roll(np.roll(mscn_patch, 1, axis=0), 1, axis=1),
        mscn_patch * np.roll(np.roll(mscn_patch, 1, axis=0), -1, axis=1),
    )
    for product in products:
        a, bl, br = estimate_aggd(product)
        eta = (br - bl) * special.gamma(2 / a) / special.gamma(1 / a)
        feat.extend([a, eta, bl, br])
    return np.asarray(feat, dtype=np.float64)


def block_features(image: Array, patch_size: int) -> Tuple[Array, Array]:
    """Return one 18-D feature vector and one sharpness value per patch."""
    mscn, sigma = mscn_and_sigma(image)
    height = mscn.shape[0] - (mscn.shape[0] % patch_size)
    width = mscn.shape[1] - (mscn.shape[1] % patch_size)
    if height < patch_size or width < patch_size:
        return np.empty((0, 18)), np.empty(0)
    mscn, sigma = mscn[:height, :width], sigma[:height, :width]
    feats, sharpness = [], []
    for y in range(0, height, patch_size):
        for x in range(0, width, patch_size):
            feats.append(feature_18(mscn[y:y + patch_size, x:x + patch_size]))
            sharpness.append(float(sigma[y:y + patch_size, x:x + patch_size].mean()))
    return np.vstack(feats), np.asarray(sharpness)


def prepare_niqe_scales(image: Array, patch_size: int) -> Tuple[Array, Array]:
    """Return the two spatial scales used by the LIVE NIQE release.

    LIVE first crops the pristine/test image to complete scale-1 blocks and
    only then downsamples that cropped image by 0.5.  Applying the crop after
    resizing changes the scale-2 samples whenever an image dimension is not a
    multiple of the 96-pixel block size.  CPU and CUDA paths call this same
    helper so their resize inputs are identical.
    """
    image = np.asarray(image)
    height = image.shape[0] - image.shape[0] % patch_size
    width = image.shape[1] - image.shape[1] % patch_size
    if height < patch_size or width < patch_size:
        return np.empty((0, 0), dtype=np.float32), np.empty((0, 0), dtype=np.float32)
    scale1 = image[:height, :width].astype(np.float32, copy=False)
    # OpenCV bicubic is deterministic and shared by CPU/GPU evaluation here.
    # It is the only backend-level difference from MATLAB's imresize; the
    # feature construction and statistical distance follow the LIVE release.
    scale2 = cv2.resize(scale1, (width // 2, height // 2), interpolation=cv2.INTER_CUBIC)
    return scale1, scale2


def niqe_features(image: Array, patch_size: int = 96) -> Tuple[Array, Array]:
    """Original NIQE: concatenate matching 18-D features over two scales."""
    patch_size = int(patch_size)
    image1, image2 = prepare_niqe_scales(image, patch_size)
    if image1.size == 0:
        return np.empty((0, 36)), np.empty(0)
    scale1, sharpness = block_features(image1, patch_size)
    scale2, _ = block_features(image2, max(8, patch_size // 2))
    count = min(len(scale1), len(scale2))
    return np.hstack([scale1[:count], scale2[:count]]), sharpness[:count]


def fit_pristine_model(images: Iterable[Array], patch_size: int, sharpness_fraction: float) -> Tuple[Array, Array, int]:
    selected: List[Array] = []
    for image in images:
        features, sharpness = niqe_features(image, patch_size)
        if not len(features):
            continue
        threshold = sharpness_fraction * float(sharpness.max())
        keep = features[sharpness > threshold]
        # Avoid an empty model for low-texture normal reference images.
        selected.append(keep if len(keep) else features[np.argmax(sharpness)][None, :])
    if not selected:
        raise ValueError("No NIQE patches could be extracted from pristine images.")
    matrix = np.vstack(selected)
    mu = matrix.mean(axis=0)
    cov = np.cov(matrix, rowvar=False)
    eps = max(float(np.trace(cov)) / cov.shape[0], 1.0) * 1e-8
    return mu, cov + eps * np.eye(cov.shape[0]), int(len(matrix))


def niqe_score(image: Array, pristine_mu: Array, pristine_cov: Array, patch_size: int) -> float:
    features, _ = niqe_features(image, patch_size)
    if len(features) < 2:
        return float("nan")
    mu = features.mean(axis=0)
    cov = np.cov(features, rowvar=False)
    avg_cov = (pristine_cov + cov) / 2.0
    delta = pristine_mu - mu
    return float(np.sqrt(delta @ linalg.pinv(avg_cov) @ delta))


def _torch_aggd(v: "torch.Tensor", gamma_grid: "torch.Tensor", r_gamma: "torch.Tensor",
                scale_grid: "torch.Tensor") -> Tuple["torch.Tensor", "torch.Tensor", "torch.Tensor"]:
    """Vectorised AGGD fit for a batch of flattened local patches."""
    v = v.reshape(v.shape[0], -1)
    left = v < 0
    right = v > 0
    left_count = left.sum(dim=1).clamp_min(1)
    right_count = right.sum(dim=1).clamp_min(1)
    left_std = torch.sqrt((v.square() * left).sum(dim=1) / left_count)
    right_std = torch.sqrt((v.square() * right).sum(dim=1) / right_count)
    gammahat = left_std / (right_std + 1e-12)
    rhat = v.abs().mean(dim=1).square() / (v.square().mean(dim=1) + 1e-12)
    rhatnorm = rhat * ((gammahat.pow(3) + 1) * (gammahat + 1)) / (gammahat.square() + 1).square()
    right_index = torch.bucketize(rhatnorm, r_gamma)
    right_index = right_index.clamp(0, len(r_gamma) - 1)
    left_index = (right_index - 1).clamp_min(0)
    choose_left = (right_index > 0) & ((rhatnorm - r_gamma[left_index]).abs() <= (r_gamma[right_index] - rhatnorm).abs())
    grid_index = torch.where(choose_left, left_index, right_index)
    alpha = gamma_grid[grid_index]
    scale = scale_grid[grid_index]
    return alpha, left_std * scale, right_std * scale


def _torch_feature18(mscn_patches: "torch.Tensor", gamma_grid: "torch.Tensor", r_gamma: "torch.Tensor",
                     scale_grid: "torch.Tensor") -> "torch.Tensor":
    alpha, beta_l, beta_r = _torch_aggd(mscn_patches, gamma_grid, r_gamma, scale_grid)
    features = [alpha, (beta_l + beta_r) / 2.0]
    products = (
        mscn_patches * torch.roll(mscn_patches, 1, dims=2),
        mscn_patches * torch.roll(mscn_patches, 1, dims=1),
        mscn_patches * torch.roll(mscn_patches, (1, 1), dims=(1, 2)),
        mscn_patches * torch.roll(mscn_patches, (1, -1), dims=(1, 2)),
    )
    for product in products:
        a, bl, br = _torch_aggd(product, gamma_grid, r_gamma, scale_grid)
        eta = (br - bl) * torch.exp(torch.lgamma(2 / a) - torch.lgamma(1 / a))
        features.extend((a, eta, bl, br))
    return torch.stack(features, dim=1)


def _torch_block_features(images: "torch.Tensor", patch_size: int, kernel: "torch.Tensor",
                          gamma_grid: "torch.Tensor", r_gamma: "torch.Tensor", scale_grid: "torch.Tensor") -> "torch.Tensor":
    """Return [batch, blocks, 18] features with nearest-boundary Gaussian MSCN."""
    padded = torch_f.pad(images, (3, 3, 3, 3), mode="replicate")
    mu = torch_f.conv2d(padded, kernel)
    sigma = torch.sqrt(torch.abs(torch_f.conv2d(torch_f.pad(images.square(), (3, 3, 3, 3), mode="replicate"), kernel) - mu.square()))
    mscn = (images - mu) / (sigma + 1.0)
    height = mscn.shape[-2] - (mscn.shape[-2] % patch_size)
    width = mscn.shape[-1] - (mscn.shape[-1] % patch_size)
    if height < patch_size or width < patch_size:
        return torch.empty((images.shape[0], 0, 18), device=images.device)
    patches = mscn[:, :, :height, :width].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    batch, _, blocks_y, blocks_x, _, _ = patches.shape
    # The reference CPU path converts every MSCN patch to float64 inside
    # ``estimate_aggd``.  Keep convolution/MSCN in float32 (matching SciPy's
    # float32 output), then promote the patch statistics to float64.  A fully
    # float32 AGGD fit is faster but can move the discrete alpha-grid choice
    # enough to perturb the final NIQE score by several percent.
    flat = patches.reshape(batch * blocks_y * blocks_x, patch_size, patch_size).to(torch.float64)
    features = _torch_feature18(flat, gamma_grid, r_gamma, scale_grid)
    return features.reshape(batch, blocks_y * blocks_x, 18)


def _torch_features_from_mscn(mscn: "torch.Tensor", patch_size: int,
                              gamma_grid: "torch.Tensor", r_gamma: "torch.Tensor",
                              scale_grid: "torch.Tensor") -> "torch.Tensor":
    """Extract block features from reference-computed MSCN fields.

    NIQE is sensitive to tiny convolution differences because AGGD shape
    parameters are selected from a discrete grid.  The public CPU reference
    uses SciPy ``ndimage.convolve``.  Passing those exact MSCN fields to CUDA
    preserves numerical equivalence while still accelerating the dominant
    per-block AGGD feature extraction.
    """
    height = mscn.shape[-2] - (mscn.shape[-2] % patch_size)
    width = mscn.shape[-1] - (mscn.shape[-1] % patch_size)
    if height < patch_size or width < patch_size:
        return torch.empty((mscn.shape[0], 0, 18), device=mscn.device, dtype=torch.float64)
    patches = mscn[:, :, :height, :width].unfold(2, patch_size, patch_size).unfold(3, patch_size, patch_size)
    batch, _, blocks_y, blocks_x, _, _ = patches.shape
    flat = patches.reshape(batch * blocks_y * blocks_x, patch_size, patch_size).to(torch.float64)
    features = _torch_feature18(flat, gamma_grid, r_gamma, scale_grid)
    return features.reshape(batch, blocks_y * blocks_x, 18)


def niqe_scores_gpu(images: Sequence[Array], pristine_mu: Array, pristine_cov: Array, patch_size: int = 96,
                    device: str = "cuda", batch_size: int = 8, mscn_workers: int = 4) -> List[float]:
    """CUDA-accelerated batched NIQE scores with the same 36-D feature design.

    Gaussian MSCN, local AGGD statistics, and two-scale feature extraction are
    batched on CUDA.  The small 36x36 covariance-distance step stays in SciPy,
    preserving the reference evaluator's stable pseudoinverse calculation.
    ``device='cuda'`` is required; callers should explicitly fall back to
    :func:`niqe_score` when CUDA is unavailable.
    """
    if torch is None or not torch.cuda.is_available() or not str(device).startswith("cuda"):
        raise RuntimeError("CUDA PyTorch is unavailable; use CPU NIQE instead.")
    if not images:
        return []
    shape = np.asarray(images[0]).shape
    if any(np.asarray(image).shape != shape for image in images):
        raise ValueError("GPU NIQE batches require same-shaped grayscale images.")
    dev = torch.device(device)
    gamma_grid = torch.as_tensor(GAMMA_GRID, dtype=torch.float64, device=dev)
    r_gamma = torch.as_tensor(R_GAMMA, dtype=torch.float64, device=dev)
    scale_grid = torch.as_tensor(AGGD_SCALE_GRID, dtype=torch.float64, device=dev)
    scores: List[float] = []
    for start in range(0, len(images), batch_size):
        prepared = [prepare_niqe_scales(image, patch_size) for image in images[start:start + batch_size]]
        if any(scale1.size == 0 for scale1, _ in prepared):
            scores.extend([float("nan")] * len(prepared)); continue
        raw1 = np.stack([scale1 for scale1, _ in prepared])
        raw2 = np.stack([scale2 for _, scale2 in prepared])
        # Compute MSCN with the same SciPy path as ``niqe_features``.  This
        # removes backend-dependent convolution drift before the discrete
        # AGGD grid lookup; GPU remains responsible for the expensive block
        # statistics over all patches.
        # SciPy's convolution releases the GIL.  Parallelize only this exact
        # reference operation; feature values and image order are unchanged.
        tasks = list(raw1) + list(raw2)
        if mscn_workers > 1 and len(tasks) > 1:
            with ThreadPoolExecutor(max_workers=min(mscn_workers, len(tasks)),
                                    thread_name_prefix="niqe-mscn") as pool:
                fields = list(pool.map(lambda image: mscn_and_sigma(image)[0], tasks))
        else:
            fields = [mscn_and_sigma(image)[0] for image in tasks]
        mscn1 = np.stack(fields[:len(raw1)])
        mscn2 = np.stack(fields[len(raw1):])
        tensor1 = torch.from_numpy(mscn1).to(dev, non_blocking=True).unsqueeze(1)
        tensor2 = torch.from_numpy(mscn2).to(dev, non_blocking=True).unsqueeze(1)
        scale1 = _torch_features_from_mscn(tensor1, patch_size, gamma_grid, r_gamma, scale_grid)
        scale2 = _torch_features_from_mscn(tensor2, max(8, patch_size // 2), gamma_grid, r_gamma, scale_grid)
        blocks = min(scale1.shape[1], scale2.shape[1])
        features = torch.cat((scale1[:, :blocks], scale2[:, :blocks]), dim=2).detach().cpu().numpy().astype(np.float64, copy=False)
        for feature in features:
            if len(feature) < 2:
                scores.append(float("nan")); continue
            mu = feature.mean(axis=0); cov = np.cov(feature, rowvar=False)
            avg_cov = (pristine_cov + cov) / 2.0; delta = pristine_mu - mu
            scores.append(float(np.sqrt(delta @ linalg.pinv(avg_cov) @ delta)))
    return scores


def summary(values: Sequence[float]) -> Dict[str, float]:
    array = np.asarray(values, dtype=float)
    array = array[np.isfinite(array)]
    return {
        "count": int(array.size),
        "mean": float(array.mean()),
        "median": float(np.median(array)),
        "std": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "min": float(array.min()),
        "max": float(array.max()),
    }


def orientation_scores(volume: Array, orientation: str, max_sections: int,
                       pristine_mu: Array, pristine_cov: Array, patch_size: int) -> List[float]:
    if orientation == "xy":
        sections = [volume[z] for z in evenly_spaced(list(range(volume.shape[0])), max_sections)]
    elif orientation == "xz":
        sections = [volume[:, y, :] for y in evenly_spaced(list(range(volume.shape[1])), max_sections)]
    elif orientation == "yz":
        sections = [volume[:, :, x] for x in evenly_spaced(list(range(volume.shape[2])), max_sections)]
    else:
        raise ValueError(orientation)
    return [niqe_score(section, pristine_mu, pristine_cov, patch_size) for section in sections]


def load_selected(mapping: Dict[int, Path], ids: Sequence[int]) -> Array:
    first = read_gray(mapping[ids[0]])
    volume = np.empty((len(ids), *first.shape), dtype=np.uint8)
    volume[0] = first
    for index, image_id in enumerate(ids[1:], start=1):
        image = read_gray(mapping[image_id])
        if image.shape != first.shape:
            raise ValueError(f"Inconsistent image shape at {mapping[image_id]}: {image.shape} != {first.shape}")
        volume[index] = image
    return volume


def evaluate(volume: Array, pristine_mu: Array, pristine_cov: Array, patch_size: int,
             max_xy: int, max_xz_yz: int) -> Tuple[Dict[str, Dict[str, float]], Dict[str, List[float]]]:
    scores = {
        "xy": orientation_scores(volume, "xy", max_xy, pristine_mu, pristine_cov, patch_size),
        "xz": orientation_scores(volume, "xz", max_xz_yz, pristine_mu, pristine_cov, patch_size),
        "yz": orientation_scores(volume, "yz", max_xz_yz, pristine_mu, pristine_cov, patch_size),
    }
    return {axis: summary(values) for axis, values in scores.items()}, scores


def main() -> None:
    parser = argparse.ArgumentParser(description="Correct two-scale domain NIQE for raw/processed EM folders")
    parser.add_argument("--raw-dir", required=True, type=Path)
    parser.add_argument("--processed-dir", required=True, type=Path)
    parser.add_argument("--normal-dir", required=True, type=Path,
                        help="Unpaired normal images used only to fit the pristine NIQE model")
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--max-z-slices", type=int, default=400)
    parser.add_argument("--max-xy", type=int, default=32)
    parser.add_argument("--max-xz-yz", type=int, default=24)
    parser.add_argument("--patch-size", type=int, default=96)
    parser.add_argument("--sharpness-fraction", type=float, default=0.75)
    args = parser.parse_args()

    raw = indexed_paths(args.raw_dir, "slice_*.png")
    processed = indexed_paths(args.processed_dir, "slice_*_deflickered.png")
    common_ids = sorted(set(raw) & set(processed))
    if not common_ids:
        raise ValueError("No slice IDs are shared by raw and processed folders.")
    ids = evenly_spaced(common_ids, args.max_z_slices)
    normal_paths = sorted(args.normal_dir.glob("*.png"), key=lambda p: p.name)
    if not normal_paths:
        raise ValueError(f"No normal PNG images found in {args.normal_dir}")

    pristine_mu, pristine_cov, patch_count = fit_pristine_model(
        (read_gray(path) for path in normal_paths), args.patch_size, args.sharpness_fraction
    )
    print(f"Pristine model: {len(normal_paths)} unpaired normal images, {patch_count} selected patches")
    print(f"Evaluating {len(ids)} uniformly sampled common Z slices (no Z interpolation)")

    raw_volume = load_selected(raw, ids)
    raw_metrics, raw_scores = evaluate(raw_volume, pristine_mu, pristine_cov, args.patch_size, args.max_xy, args.max_xz_yz)
    del raw_volume
    processed_volume = load_selected(processed, ids)
    processed_metrics, processed_scores = evaluate(processed_volume, pristine_mu, pristine_cov, args.patch_size, args.max_xy, args.max_xz_yz)

    result = {
        "metric": "EM-domain calibrated NIQE (not canonical natural-image NIQE)",
        "algorithm": "original two-scale 36-D NSS feature structure with sharp-patch pristine-model fitting",
        "pristine_model": {
            "normal_dir": str(args.normal_dir), "normal_image_count": len(normal_paths),
            "selected_patch_count": patch_count, "sharpness_fraction": args.sharpness_fraction,
        },
        "sampling": {"common_raw_processed_slices": len(common_ids), "evaluated_z_slices": len(ids),
                     "z_slice_ids": ids, "z_interpolation": "none"},
        "metrics": {"raw": raw_metrics, "processed": processed_metrics},
    }
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "niqe_domain_summary.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    with (args.output_dir / "niqe_domain_sections.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["volume", "orientation", "section_rank", "niqe"])
        for label, score_map in (("raw", raw_scores), ("processed", processed_scores)):
            for orientation, values in score_map.items():
                writer.writerows([[label, orientation, rank, value] for rank, value in enumerate(values)])
    for axis in ("xy", "xz", "yz"):
        before, after = raw_metrics[axis]["mean"], processed_metrics[axis]["mean"]
        print(f"{axis.upper()} NIQE: {before:.4f} -> {after:.4f} ({after - before:+.4f}; lower is better)")


if __name__ == "__main__":
    main()
