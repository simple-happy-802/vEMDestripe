"""Axis-specific clean-kidney-reference NIQE for the Bumbarger13 demo.

The MSCN and two-scale AGGD features match the verified microscopy evaluator.
The bundled NPZ contains only the frozen mean and covariance per axis.
"""
from hashlib import sha256
from pathlib import Path

import numpy as np
import pandas as pd
from numba import njit
from scipy import linalg, ndimage, special

# Bumbarger13 demo source voxel spacing in nanometres, ordered X, Y, Z.
SPACING_XYZ_NM = (3.9, 3.9, 50.0)
Z_RATIOS = {"xy": 1.0, "xz": SPACING_XYZ_NM[2] / SPACING_XYZ_NM[0],
            "yz": SPACING_XYZ_NM[2] / SPACING_XYZ_NM[1]}
AXES = tuple(Z_RATIOS)
PATCH = 96
PRIOR_SHA256 = "c5c1ec708d32900f9288b0aa9bfa46e437a76251cdfd594c75fe2254748d8d38"
REFERENCE_DOI = "10.25378/janelia.16913035.v1"

GAMMA = np.arange(0.2, 10.001, 0.001)
RATIO = special.gamma(2/GAMMA)**2 / (special.gamma(1/GAMMA) * special.gamma(3/GAMMA))
SCALE = np.sqrt(special.gamma(1/GAMMA) / special.gamma(3/GAMMA))
ETA = special.gamma(2/GAMMA) / special.gamma(1/GAMMA)
window = np.exp(-np.arange(-3, 4, dtype=float)**2 / (2 * (7/6)**2))
GAUSSIAN = np.outer(window, window)
GAUSSIAN /= GAUSSIAN.sum()
x = np.abs((np.arange(8) - 3.5) / 2)
HALF_SCALE = np.where(x <= 1, 1.5*x**3 - 2.5*x*x + 1, -0.5*x**3 + 2.5*x*x - 4*x + 2)
HALF_SCALE /= HALF_SCALE.sum()


@njit(cache=True)
def estimate(x, gamma, ratios, scales, eta_scales):
    total = absolute = left = right = 0.0
    nl = nr = 0
    for i in range(x.size):
        value = np.float64(x[i])
        square = value * value
        total += square
        absolute += abs(value)
        if value < 0:
            left += square
            nl += 1
        if value > 0:
            right += square
            nr += 1
    left_std = np.sqrt(left / max(1, nl))
    right_std = np.sqrt(right / max(1, nr))
    if left_std == 0 and right_std == 0:
        return 1.0, 0.0, 0.0, 0.0
    g = left_std / (right_std + 1e-12)
    r = (absolute / x.size)**2 / (total / x.size + 1e-12)
    moment_ratio = r * ((g**3 + 1) * (g + 1)) / (g*g + 1)**2
    hi = min(max(np.searchsorted(ratios, moment_ratio), 0), len(ratios) - 1)
    lo = max(0, hi - 1)
    pos = lo if abs(ratios[lo] - moment_ratio) <= abs(ratios[hi] - moment_ratio) else hi
    beta_left = left_std * scales[pos]
    beta_right = right_std * scales[pos]
    return gamma[pos], beta_left, beta_right, (beta_right - beta_left) * eta_scales[pos]

@njit(cache=True)
def patch_features(mscn,patch,gamma,ratios,scales,eta_scales):
    ny, nx = mscn.shape[0] // patch, mscn.shape[1] // patch
    out = np.empty((ny * nx, 18), np.float64)
    shifts = ((0, 1), (1, 0), (1, 1), (1, -1))
    vector = np.empty(patch * patch, np.float64)
    product = np.empty_like(vector)
    for iy in range(ny):
        for ix in range(nx):
            for y in range(patch):
                for x in range(patch):
                    vector[y*patch + x] = mscn[iy*patch + y, ix*patch + x]
            idx = iy*nx + ix
            a, bl, br, _ = estimate(vector, gamma, ratios, scales, eta_scales)
            out[idx, 0] = a
            out[idx, 1] = (bl + br) / 2
            for k, (dy, dx) in enumerate(shifts):
                for y in range(patch):
                    for x in range(patch):
                        neighbor = ((y - dy) % patch)*patch + (x - dx) % patch
                        product[y*patch + x] = vector[y*patch + x] * vector[neighbor]
                a, bl, br, eta = estimate(product, gamma, ratios, scales, eta_scales)
                out[idx, 2 + 4*k] = a
                out[idx, 3 + 4*k] = eta
                out[idx, 4 + 4*k] = bl
                out[idx, 5 + 4*k] = br
    return out


def _mscn(image):
    mean = ndimage.convolve(image, GAUSSIAN, mode="nearest")
    variance = ndimage.convolve(image * image, GAUSSIAN, mode="nearest") - mean * mean
    return (image - mean) / (np.sqrt(np.abs(variance)) + 1)


def patch_matrix(plane):
    """Extract each patch's 36 NIQE features from a grey plane on [0,255]."""
    height, width = (size // PATCH * PATCH for size in plane.shape)
    if not height or not width:
        return np.empty((0, 36))
    image = np.asarray(plane[:height, :width], dtype=np.float64)
    if not np.isfinite(image).all() or image.min() < 0 or image.max() > 255:
        raise ValueError("NIQE requires finite grey values on [0,255]")
    filtered = ndimage.correlate1d(image, HALF_SCALE, axis=0, mode="reflect", origin=-1)
    half = ndimage.correlate1d(filtered[::2], HALF_SCALE, axis=1, mode="reflect", origin=-1)[:, ::2]
    return np.hstack([
        patch_features(_mscn(scale), block, GAMMA, RATIO, SCALE, ETA)
        for scale, block in ((image, PATCH), (half, PATCH // 2))
    ])


def niqe_distance(features, mean, covariance):
    if len(features) < 2:
        return None
    delta = features.mean(0) - mean
    average_covariance = (covariance + np.cov(features, rowvar=False)) / 2
    squared = float(delta @ linalg.pinv(average_covariance) @ delta)
    if not np.isfinite(squared) or squared < -1e-9:
        raise ValueError("Invalid NIQE distance")
    return float(np.sqrt(max(squared, 0.0)))


def load_prior():
    path = Path(__file__).with_name("clean_kidney_niqe_prior.npz")
    if sha256(path.read_bytes()).hexdigest() != PRIOR_SHA256:
        raise ValueError("Clean-kidney NIQE prior checksum mismatch")
    with np.load(path, allow_pickle=False) as f:
        return {axis: (f[axis + "_mu"], f[axis + "_cov"]) for axis in AXES}


def demo_plane(volume, axis, index, margin=0):
    """Shared XY/XZ/YZ geometry for display and evaluation."""
    if axis == "xy":
        return volume[index]
    if axis not in Z_RATIOS:
        raise ValueError("Expected an XY, XZ or YZ plane")
    plane = volume[:, index, :] if axis == "xz" else volume[:, :, index]
    if margin:
        plane = plane[:, margin:-margin]
    return ndimage.zoom(plane.astype(np.float32), (Z_RATIOS[axis], 1),
                        order=1, grid_mode=False)


def plot_demo_side(volumes, axis, position):
    from matplotlib import pyplot as plt

    fig, panels = plt.subplots(1, len(volumes), figsize=(19, 3), layout="constrained")
    for panel, (method, volume) in zip(panels, volumes.items()):
        panel.imshow(demo_plane(volume, axis, position), cmap="gray", vmin=0, vmax=255,
                     aspect="equal", interpolation="nearest")
        panel.set_title(method)
        panel.axis("off")
    fig.suptitle(f"{axis.upper()} at {'Y' if axis == 'xz' else 'X'}={position} · "
                 f"physical Z reslicing {Z_RATIOS[axis]:.2f}×")
    plt.show()


def _indices(start, stop, count):
    return np.unique(np.linspace(start, stop - 1, min(count, stop - start)).round().astype(int))


def evaluate_demo(volumes, section_ids, *, xy_samples=8, side_samples=6, margin=32):
    """Score matched saved uint8 volumes, then return a summary and per-plane data."""
    prior = load_prior()
    shape = next(iter(volumes.values())).shape
    if len(shape) != 3 or shape[0] < 2 or len(section_ids) != shape[0]:
        raise ValueError("Expected numbered, matched ZYX demo images")
    if min(xy_samples, side_samples) < 1 or margin < 0 or min(shape[1:]) <= 2 * margin:
        raise ValueError("Invalid metric sampling parameters")
    positions = {"xy": _indices(0, shape[0], xy_samples),
                 "xz": _indices(margin, shape[1] - margin, side_samples),
                 "yz": _indices(margin, shape[2] - margin, side_samples)}
    summary, detail = [], []
    for method, volume in volumes.items():
        if volume.shape != shape or volume.dtype != np.uint8:
            raise ValueError(f"{method}: expected the matched saved uint8 volume")
        row = {"method": method}
        for axis, indices in positions.items():
            scores = []
            for index in indices:
                plane = demo_plane(volume, axis, index, margin if axis != "xy" else 0)
                features = patch_matrix(plane)
                score = niqe_distance(features, *prior[axis])
                if score is not None:
                    scores.append(score)
                detail.append({
                    "method": method,
                    "axis": axis.upper(),
                    "section_position": section_ids[index] if axis == "xy" else int(index),
                    "fixed_axis": {"xy": "Z", "xz": "Y", "yz": "X"}[axis],
                    "native_z_first": section_ids[0],
                    "native_z_last": section_ids[-1],
                    "z_ratio": Z_RATIOS[axis],
                    "lateral_margin": margin if axis != "xy" else 0,
                    "height_px": plane.shape[0],
                    "width_px": plane.shape[1],
                    "patch_count": len(features),
                    "few_patches": len(features) < 37,
                    "niqe_clean_kidney": score,
                })
            row[f"NIQE_{axis.upper()}"] = float(np.mean(scores)) if scores else None
        deltas = np.diff(volume.mean(axis=(1, 2), dtype=np.float64))
        row["ASID"] = float(np.median(np.abs(deltas - np.median(deltas))))
        summary.append(row)
    return pd.DataFrame(summary).set_index("method"), pd.DataFrame(detail)
