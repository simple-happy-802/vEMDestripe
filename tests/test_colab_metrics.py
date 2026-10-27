"""Small alignment and photometric tests; no full NIQE evaluation required."""

from pathlib import Path

import numpy as np
import pytest
from PIL import Image

from colab.demo_metrics import (
    METHOD_ORDER,
    asid_z_mean_mad_diff,
    global_affine_match,
    load_demo_volumes,
    z_brightness_hf_percent,
)


def _write_slices(folder: Path, shape=(4, 6), indices=(7, 8)) -> None:
    folder.mkdir(parents=True, exist_ok=True)
    for index in indices:
        image = np.full(shape, index, dtype=np.uint8)
        Image.fromarray(image).save(folder / f"slice_{index:05d}.png")


def _demo_layout(tmp_path):
    root = tmp_path / "assets"
    emisac = tmp_path / "official_emisac"
    folders = {
        "Original": root / "input",
        "EMISAC": emisac,
        **{method: root / "precomputed" / method for method in METHOD_ORDER[2:]},
    }
    for folder in folders.values():
        _write_slices(folder)
    return root, emisac, folders


def test_demo_loader_accepts_aligned_volumes(tmp_path):
    root, emisac, _ = _demo_layout(tmp_path)
    volumes, indices = load_demo_volumes(root, emisac)
    assert tuple(volumes) == METHOD_ORDER
    assert indices == [7, 8]
    assert all(volume.shape == (2, 4, 6) for volume in volumes.values())
    assert all(volume.dtype == np.uint8 for volume in volumes.values())


def test_demo_loader_rejects_mismatched_section_indices(tmp_path):
    root, emisac, folders = _demo_layout(tmp_path)
    old = folders["FECNet"] / "slice_00008.png"
    old.rename(old.with_name("slice_00009.png"))
    with pytest.raises(ValueError, match="FECNet indices do not match Original"):
        load_demo_volumes(root, emisac)


def test_demo_loader_rejects_mismatched_zyx_shape(tmp_path):
    root, emisac, folders = _demo_layout(tmp_path)
    _write_slices(folders["SSCOR"], shape=(6, 4))
    with pytest.raises(ValueError, match="SSCOR shape .* does not match Original"):
        load_demo_volumes(root, emisac)


def test_demo_loader_requires_official_emisac_output(tmp_path):
    with pytest.raises(FileNotFoundError, match="Official EMISAC output directory is missing"):
        load_demo_volumes(tmp_path / "assets", tmp_path / "missing_official_output")


def test_global_affine_match_recovers_one_scale_and_bias():
    source = np.arange(12, dtype=np.uint8).reshape(2, 2, 3)
    target = source * 2 + 5
    matched, scale, bias = global_affine_match(source, target)
    np.testing.assert_array_equal(matched, target)
    assert scale == pytest.approx(2.0)
    assert bias == pytest.approx(5.0)


def test_global_affine_match_rejects_zero_contrast():
    source = np.ones((2, 4, 6), dtype=np.uint8)
    with pytest.raises(ValueError, match="zero-contrast"):
        global_affine_match(source, source)


def test_native_z_profile_diagnostics():
    assert asid_z_mean_mad_diff([0, 2, 6, 12]) == pytest.approx(2.0)
    assert asid_z_mean_mad_diff([5]) == 0.0
    assert z_brightness_hf_percent([5, 5, 5, 5]) == pytest.approx(0.0)
