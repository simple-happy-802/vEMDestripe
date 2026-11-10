import hashlib

import cv2
import numpy as np
import pytest

from vemdestripe.optimized_alignment import fit_volume_luts


def test_luts_match_frozen_reference_with_fallback_and_gain_limits():
    # Frozen from the original fit_scheme(..., "trend_both_s16"). This volume
    # includes six constant sections and nine gain-limited sections.
    depth = np.arange(97)[:, None]
    pixel = np.arange(256)[None, :]
    center = 70 + depth % 47
    width = np.where(depth % 17 == 0, 1, 5 + depth % 29)
    source = np.maximum(0, width - np.abs(pixel - center))
    inputs = np.maximum(0, 46 - np.abs(pixel - (88 + depth % 11)))
    source = source / source.sum(axis=1, keepdims=True)
    inputs = inputs / inputs.sum(axis=1, keepdims=True)
    original_source = source.copy()
    original_inputs = inputs.copy()

    luts = fit_volume_luts(source, inputs)

    assert luts.shape == (97, 256)
    assert luts.dtype == np.uint8
    assert hashlib.sha256(luts.tobytes()).hexdigest() == (
        "f163a827ce6a0f3fb56fd0161ce6ed9ab899eb5e9acf9914833876919266cae4"
    )
    assert np.all(np.diff(luts.astype(np.int16), axis=1) >= 0)
    np.testing.assert_array_equal(source, original_source)
    np.testing.assert_array_equal(inputs, original_inputs)
    image = np.arange(256, dtype=np.uint8).reshape(16, 16)
    for lut in luts:
        np.testing.assert_array_equal(cv2.LUT(image, lut), lut[image])


def test_single_section_lut_applies_exact_byte_mapping():
    source = np.zeros((1, 256), np.float64)
    inputs = np.zeros_like(source)
    source[0, :64] = 1 / 64
    inputs[0, 64:128] = 1 / 64

    lut = fit_volume_luts(source, inputs)[0]

    expected = np.clip(np.arange(256) + 64, 0, 255).astype(np.uint8)
    np.testing.assert_array_equal(lut, expected)


@pytest.mark.parametrize(
    "invalid",
    [
        np.ones(256),
        np.ones((2, 255)),
        np.empty((0, 256)),
        np.full((2, 256), -1.0),
        np.full((2, 256), np.nan),
        np.full((2, 256), np.inf),
        np.zeros((2, 256)),
    ],
)
@pytest.mark.parametrize("argument", ["source", "input"])
def test_invalid_histograms_are_rejected(invalid, argument):
    valid = np.ones((2, 256))
    source, inputs = (invalid, valid) if argument == "source" else (valid, invalid)
    with pytest.raises(ValueError):
        fit_volume_luts(source, inputs)


def test_source_sections_require_equal_positive_mass():
    inputs = np.ones((2, 256))
    source = inputs.copy()
    source[1] *= 2
    with pytest.raises(ValueError, match="Equal positive"):
        fit_volume_luts(source, inputs)
    source[1] = 0
    with pytest.raises(ValueError, match="Equal positive"):
        fit_volume_luts(source, inputs)


def test_constant_distributions_are_rejected():
    constant = np.zeros((2, 256))
    constant[:, 80] = 1
    valid = np.ones_like(constant)
    with pytest.raises(ValueError, match="constant source"):
        fit_volume_luts(constant, valid)
    with pytest.raises(ValueError, match="Nonconstant target"):
        fit_volume_luts(valid, constant)
    narrow_target = constant.copy()
    narrow_target[:, 80] = 0.99
    narrow_target[:, 81] = 0.01
    with pytest.raises(ValueError, match="Nonconstant target"):
        fit_volume_luts(valid, narrow_target)
