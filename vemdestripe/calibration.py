"""Whole-volume photometric calibration in the vEMDestripe output pipeline.

This is not image registration or a per-section contrast enhancement. One
mean/std affine map is fitted from *all saved uint8 network-output pixels* to
the pooled pixels of the selected original reference snapshots. The same
256-entry lookup table is then applied to every output section. Baseline
methods do not call this module.

The histogram arithmetic and float32 lookup-table quantization intentionally
match the user's audited ``common.py`` implementation. In particular, no
fused or block-averaged reference image is used to estimate target contrast.
"""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np
from PIL import Image


def moments_from_histograms(hist: np.ndarray) -> tuple[float, float]:
    """Population moments of the mean of normalized, equal-size slice histograms.

    This expression is deliberately identical to the audited paper evaluator.
    For original-reference pooling, pass one normalized pooled histogram.
    """
    hist = np.asarray(hist, dtype=np.float64)
    if hist.ndim != 2 or hist.shape[1] != 256 or not len(hist):
        raise ValueError('Expected a nonempty (sections, 256) histogram array')
    if not np.isfinite(hist).all() or (hist < 0).any():
        raise ValueError('Histogram values must be finite and nonnegative')
    if not np.allclose(hist.sum(axis=1), 1.0, atol=1e-10, rtol=0):
        raise ValueError('Each histogram must sum to one')
    p = hist.mean(axis=0)
    levels = np.arange(256, dtype=float)
    mu = float(p @ levels)
    return mu, float(np.sqrt(max(0, p @ (levels * levels) - mu * mu)))


def fit_affine(source_mean_std: Sequence[float], target_mean_std: Sequence[float]) -> tuple[float, float]:
    """Fit one mean/std map, with the audited constant-volume behavior."""
    mu, sd = map(float, source_mean_std)
    tm, ts = map(float, target_mean_std)
    if not np.isfinite([mu, sd, tm, ts]).all() or sd < 0 or ts < 0:
        raise ValueError('Means/stds must be finite and stds nonnegative')
    if sd <= 1e-8:
        if ts > 1e-8:
            raise ValueError('Cannot match a constant volume')
        return 1., tm - mu
    scale = ts / sd
    return scale, tm - scale * mu


def calibration_lut(scale: float, bias: float) -> np.ndarray:
    """Exact float32 256-level LUT: round(clip(value*scale+bias, 0, 255)).

    Do not replace this with float64 per-pixel arithmetic: values near a
    half-integer can otherwise round differently from the recorded pipeline.
    """
    if not np.isfinite([scale, bias]).all() or scale < 0:
        raise ValueError('A finite nonnegative scale and finite bias are required')
    return np.rint(np.clip(np.arange(256, dtype=np.float32) * scale + bias, 0, 255)).astype(np.uint8)


def _uint8_section(image: np.ndarray) -> np.ndarray:
    image = np.asarray(image)
    if image.dtype != np.uint8 or image.ndim != 2 or image.size == 0:
        raise ValueError('Calibration requires nonempty grayscale uint8 sections')
    return image


def volume_statistics(images: Iterable[np.ndarray]) -> dict:
    """Measure every voxel of an equal-size saved-output volume, without crops."""
    rows, shape = [], None
    for image in images:
        image = _uint8_section(image)
        if shape is None:
            shape = image.shape
        if image.shape != shape:
            raise ValueError('All output sections must have the same shape')
        rows.append(np.bincount(image.ravel(), minlength=256) / image.size)
    if not rows:
        raise ValueError('The output volume has no sections')
    mean, std = moments_from_histograms(np.stack(rows))
    return {'mean': mean, 'std': std, 'section_count': len(rows),
            'pixel_count': int(len(rows) * np.prod(shape)), 'shape_yx': list(shape),
            'aggregation': 'all saved output pixels; mean of equal-size normalized slice histograms; population std'}


def reference_statistics(images: Iterable[np.ndarray]) -> dict:
    """Pool original snapshot pixels before moments; never use fused reference."""
    histogram = np.zeros(256, np.float64)
    count = 0
    for image in images:
        image = _uint8_section(image)
        histogram += np.bincount(image.ravel(), minlength=256)
        count += 1
    total = int(histogram.sum())
    if not count or not total:
        raise ValueError('No original reference pixels are available')
    histogram /= histogram.sum()
    mean, std = moments_from_histograms(histogram[None, :])
    return {'mean': mean, 'std': std, 'section_count': count, 'pixel_count': total,
            'histogram': histogram.tolist(),
            'aggregation': 'pool all pixels of selected original snapshots before population moments'}


def calibration_record(source_statistics: dict, target_statistics: dict,
                       source_scope: str = 'all saved output voxels',
                       target_source: str = 'pooled_original_reference_snapshots') -> dict:
    scale, bias = fit_affine((source_statistics['mean'], source_statistics['std']),
                             (target_statistics['mean'], target_statistics['std']))
    return {'schema_version': 1, 'method': 'vEMDestripe',
            'algorithm': 'whole_volume_mean_std_reference_calibration',
            'source_scope': source_scope, 'target_source': target_source,
            'source_statistics': dict(source_statistics), 'target_statistics': dict(target_statistics),
            'scale': scale, 'bias': bias, 'lut': calibration_lut(scale, bias).tolist(),
            'formula': 'rint(clip(float32_uint8_level * scale + bias, 0, 255)).astype(uint8)',
            'population_std': True, 'per_slice_fit': False,
            'fused_or_block_averaged_reference_used_as_target': False}


def apply_calibration(array: np.ndarray, scale: float, bias: float) -> np.ndarray:
    array = np.asarray(array)
    if array.dtype != np.uint8:
        raise ValueError('Only saved uint8 gray units are accepted; no implicit rescaling')
    return calibration_lut(scale, bias)[array]


def read_uint8_image(path: str | Path) -> np.ndarray:
    with Image.open(path) as image:
        result = np.asarray(image).copy()
    return _uint8_section(result)


def sha256_file(path: str | Path) -> str:
    value = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def calibrate_directory(input_dir: str | Path, output_dir: str | Path,
                        reference_paths: Sequence[str | Path] | None = None,
                        *, input_pattern: str = '*_deflickered.png',
                        write_patterns: Sequence[str] | None = None,
                        expected_count: int | None = None,
                        target_statistics: dict | None = None,
                        target_source: str = 'pooled_original_reference_snapshots') -> dict:
    """Fit from a complete saved directory, write new final images and metadata.

    Two passes keep memory independent of image-volume size. ``input_pattern``
    selects one representation for statistics; ``write_patterns`` can include
    PNG and TIFF copies without counting the same image twice. Existing final
    images/metadata are never overwritten. Raw inputs remain immutable.
    """
    source, destination = Path(input_dir).resolve(), Path(output_dir).resolve()
    if source == destination or destination.is_relative_to(source):
        raise ValueError('Final output must be separate from the raw input directory')
    paths = sorted(source.glob(input_pattern))
    if not paths:
        raise ValueError(f'No raw output files match {input_pattern!r}')
    if expected_count is not None and len(paths) != expected_count:
        raise ValueError(f'Incomplete network output: expected {expected_count}, found {len(paths)}')
    if len({p.name for p in paths}) != len(paths):
        raise ValueError('Repeated source filename')
    source_stats = volume_statistics(read_uint8_image(path) for path in paths)
    if target_statistics is None:
        if not reference_paths:
            raise ValueError('Original reference snapshots are required for calibration')
        references = [Path(p) for p in reference_paths]
        if len({p.resolve() for p in references}) != len(references):
            raise ValueError('Original reference paths must be unique')
        target_statistics = reference_statistics(read_uint8_image(path) for path in references)
        reference_files = [{'name': p.name, 'sha256': sha256_file(p)} for p in references]
    else:
        if reference_paths:
            raise ValueError('Supply original reference paths or explicit target statistics, not both')
        reference_files = []
    record = calibration_record(source_stats, target_statistics, target_source=target_source)
    record['reference_files'] = reference_files
    record['input_pattern'] = input_pattern
    record['raw_source_files'] = [{'name': p.name, 'sha256': sha256_file(p)} for p in paths]
    to_write = {}
    for pattern in write_patterns or (input_pattern,):
        for path in sorted(source.glob(pattern)):
            to_write[path.name] = path
    if not to_write:
        raise ValueError('No images selected for final output')
    destination.mkdir(parents=True, exist_ok=True)
    metadata_path = destination / 'CALIBRATION.json'
    if metadata_path.exists() or any((destination / name).exists() for name in to_write):
        raise FileExistsError('Final output images or CALIBRATION.json already exist; choose a new output directory')
    lut = np.asarray(record['lut'], dtype=np.uint8)
    outputs = []
    for name, path in to_write.items():
        image = read_uint8_image(path)
        if tuple(image.shape) != tuple(source_stats['shape_yx']):
            raise ValueError(f'Output geometry differs from fitted source volume: {name}')
        final = destination / name
        Image.fromarray(lut[image]).save(final)
        outputs.append({'name': name, 'sha256': sha256_file(final)})
    record.update(complete=True, output_files=outputs,
                  output_note='Final images are saved after vEMDestripe-only calibration; evaluate these files, not network_raw/')
    metadata_path.write_text(json.dumps(record, indent=2, ensure_ascii=False) + '\n', encoding='utf-8')
    return record
