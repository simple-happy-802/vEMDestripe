import numpy as np

from vemdestripe.reference import block_average, build_reference


def test_float_mean_then_block_average():
    first = np.arange(16, dtype=np.uint8).reshape(4, 4)
    second = first + 2
    result = build_reference([first, second], block_size=2, quantize=False)
    expected = block_average((first.astype(np.float32) + second) / 2.0, 2)
    np.testing.assert_allclose(result, expected, rtol=0, atol=0)
    assert np.unique(result[:2, :2]).size == 1


def test_partial_edge_blocks_are_filled():
    image = np.arange(35, dtype=np.float32).reshape(5, 7)
    result = block_average(image, 4)
    assert result.shape == image.shape
    assert np.isfinite(result).all()
    assert np.all(result[4:, 4:] == image[4:, 4:].mean())

