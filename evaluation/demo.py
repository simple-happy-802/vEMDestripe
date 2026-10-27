"""Read saved final method images and score them without photometric fitting.

vEMDestripe calibration belongs to inference/export, never this evaluator.
NIQE's intrinsic feature construction is part of that metric, not a change
to the saved restoration images. No method's image values are changed here.
"""
from pathlib import Path
from typing import Mapping
import numpy as np
import pandas as pd

try:
    from colab import demo_metrics as metrics
except ImportError:
    import demo_metrics as metrics


def load_saved_methods(method_directories: Mapping[str, str | Path],
                       expected_ids=None) -> tuple[dict, list[int]]:
    volumes, reference_ids, reference_shape = {}, None, None
    for method in metrics.METHOD_ORDER:
        volume, ids = metrics.load_png_volume(method_directories[method])
        if reference_ids is None:
            reference_ids, reference_shape = ids, volume.shape
        if ids != reference_ids or volume.shape != reference_shape:
            raise ValueError(f'{method}: saved sections or shape are not aligned')
        if expected_ids is not None and ids != list(expected_ids):
            raise ValueError(f'{method}: saved section IDs differ from the declared window')
        volumes[method] = volume
    return volumes, reference_ids


def score_volumes(volumes: Mapping[str, np.ndarray], niqe_model: str | Path,
                  *, xy_samples=8, cross_sections=6, z_ratio=6.0, border=32) -> pd.DataFrame:
    """Pure measurement of the supplied saved pixel values; no rescaling fit."""
    if xy_samples < 1 or cross_sections < 1 or z_ratio <= 0 or border < 0:
        raise ValueError('Invalid metric sampling parameters')
    with np.load(niqe_model) as model:
        mu, cov = model['mu'], model['cov']
    rows = []
    for method in metrics.METHOD_ORDER:
        volume = volumes[method]
        if volume.dtype != np.uint8 or volume.ndim != 3 or len(volume) < 2:
            raise ValueError('Evaluator requires saved uint8 ZYX images, not a rescaled array')
        if min(volume.shape[1:]) <= 2 * border:
            raise ValueError('Border exceeds the saved image geometry')
        indices = metrics._spaced(len(volume), xy_samples)
        xy = [metrics.niqe_score(volume[index], mu, cov, 96) for index in indices]
        finite = [value for value in xy if np.isfinite(value)]
        profile = volume.mean(axis=(1, 2), dtype=np.float64)
        rows.append({
            'method': method, 'input_stage': 'saved_final_output',
            'evaluator_photometric_transform': False, 'demo_sections': len(volume),
            'ASID_MAD': metrics.asid_z_mean_mad_diff(profile),
            'local_ASID': metrics.local_asid(volume),
            'Z_HF_percent': metrics.z_brightness_hf_percent(profile),
            'NIQE_XY': float(np.mean(finite)) if finite else float('nan'),
            'NIQE_XZ': metrics._niqe_orientation(volume, 'xz', cross_sections, mu, cov, 96, z_ratio, border),
            'NIQE_YZ': metrics._niqe_orientation(volume, 'yz', cross_sections, mu, cov, 96, z_ratio, border),
            'contrast_index': float(np.mean([metrics.contrast_index(volume[i]) for i in indices])),
            'blur_index': float(np.mean([metrics.blur_index(volume[i]) for i in indices])),
            'xy_samples': len(indices), 'cross_sections': cross_sections,
            'cross_plane_niqe_z_ratio': z_ratio})
    return pd.DataFrame(rows).set_index('method').loc[list(metrics.METHOD_ORDER)]


def evaluate_saved_demo(method_directories, niqe_model, output_csv, **kwargs):
    """Reload exported files, compute diagnostics, and write a table only."""
    volumes, _ = load_saved_methods(method_directories)
    table = score_volumes(volumes, niqe_model, **kwargs)
    output = Path(output_csv)
    output.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(output)
    return table
