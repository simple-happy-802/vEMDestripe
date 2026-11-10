"""Volume postprocessing with one byte lookup table per section."""
from __future__ import annotations

import numpy as np
from scipy.ndimage import gaussian_filter1d


_EPS = 1e-8
_LEVELS = np.arange(256, dtype=np.float64)


def _histograms(values: np.ndarray) -> np.ndarray:
    histograms = np.asarray(values, dtype=np.float64)
    if (
        histograms.ndim != 2
        or histograms.shape[1] != 256
        or not len(histograms)
        or not np.isfinite(histograms).all()
        or np.any(histograms < 0)
    ):
        raise ValueError("Expected finite nonnegative nonempty N x 256 histograms")
    return histograms


def _pooled_statistics(histograms: np.ndarray) -> tuple[float, float, float]:
    histogram = histograms.sum(axis=0)
    mass = histogram.sum()
    if not np.isfinite(histogram).all() or not np.isfinite(mass) or mass <= 0:
        raise ValueError("Expected positive finite histogram mass")
    probabilities = histogram / mass
    mean = float(probabilities @ _LEVELS)
    variance = probabilities @ (_LEVELS * _LEVELS) - mean * mean
    std = float(np.sqrt(max(0, variance)))
    cumulative = np.cumsum(probabilities)
    lower = float(np.searchsorted(cumulative, 0.25, side="left"))
    upper = float(np.searchsorted(cumulative, 0.75, side="left"))
    return mean, std, upper - lower


def fit_volume_luts(
    source_histograms: np.ndarray, input_histograms: np.ndarray
) -> np.ndarray:
    """Fit the volume postprocessing LUTs from ordered section histograms.

    ``source_histograms`` contains the saved uint8 network-output histograms.
    Its rows must have equal positive mass. ``input_histograms`` contains the
    original acquired input histograms, pooled across all rows for the target.
    The inference pipeline uses float64 counts divided by each section's size.

    This preserves the ``trend_both_s16`` fit, including reflected Gaussian
    boundaries, constant-section fallback, global anchoring, gain limits and
    float32 quantization. The result has shape ``(sections, 256)`` and dtype
    ``uint8``. Apply a section's mapping by indexing ``luts[z][image]``.
    """
    source = _histograms(source_histograms)
    inputs = _histograms(input_histograms)
    masses = source.sum(axis=1)
    if np.any(masses <= 0) or not np.allclose(
        masses, masses[0], rtol=1e-12, atol=1e-12
    ):
        raise ValueError("Equal positive per-section pixel mass required")

    target_mean, target_std, target_iqr = _pooled_statistics(inputs)
    source_mean, source_std, _ = _pooled_statistics(source)
    if target_std <= _EPS or target_iqr <= _EPS:
        raise ValueError("Nonconstant target distribution required")
    if source_std <= _EPS:
        raise ValueError("Cannot create target contrast from a constant source")
    global_scale = target_std / source_std
    global_bias = target_mean - global_scale * source_mean

    probabilities = source / source.sum(axis=1, keepdims=True)
    means = probabilities @ _LEVELS
    variances = probabilities @ (_LEVELS * _LEVELS) - means * means
    stds = np.sqrt(np.maximum(0, variances))
    stable_stds = np.where(stds > _EPS, stds, source_std)
    scales = np.exp(
        gaussian_filter1d(np.log(stable_stds), 16.0, mode="reflect", truncate=4.0)
    ) / stable_stds
    biases = gaussian_filter1d(means, 16.0, mode="reflect", truncate=4.0) - scales * means
    constant_sections = stds <= _EPS
    scales[constant_sections] = global_scale
    biases[constant_sections] = global_bias

    centers = scales * means + biases
    pooled_mean = float(centers.mean())
    pooled_variance = float(
        np.mean(scales * scales * stds**2 + centers**2) - pooled_mean * pooled_mean
    )
    pooled_std = float(np.sqrt(max(0, pooled_variance)))
    if pooled_std <= _EPS:
        raise ValueError("Cannot anchor a constant volume to nonconstant contrast")
    gain = target_std / pooled_std
    offset = target_mean - gain * pooled_mean
    scales *= gain
    biases = biases * gain + offset
    scales = np.clip(scales, global_scale * 0.25, global_scale * 4)
    biases += target_mean - float(np.mean(scales * means + biases))
    if not np.isfinite(scales).all() or not np.isfinite(biases).all() or np.any(scales <= 0):
        raise ValueError("Expected finite positive affine mapping")

    # Preserve the reference multiply/add precision before rounding to bytes.
    unbounded = (
        _LEVELS.astype(np.float32)[None, :] * scales.astype(np.float32)[:, None]
        + biases.astype(np.float32)[:, None]
    )
    return np.rint(np.clip(unbounded, 0, 255)).astype(np.uint8)
