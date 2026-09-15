import numpy as np
import torch

from infer import VEMDestripeInferencer


class SelectInput(torch.nn.Module):
    def forward(self, value):
        return value[:, :1]


def test_tiled_blend_has_no_zero_weight_border():
    engine = VEMDestripeInferencer.__new__(VEMDestripeInferencer)
    engine.inference_size = 8
    engine.overlap_ratio = 0.25
    engine.disable_progress_bar = True
    engine.device = torch.device("cpu")
    engine.model = SelectInput()
    image = np.random.default_rng(42).random((13, 11), dtype=np.float32)
    reference = np.zeros_like(image)
    output = engine._sliding_window_inference(image, reference)
    np.testing.assert_allclose(output, image, rtol=0, atol=2e-7)
    assert np.isfinite(output[[0, -1], :]).all()
    assert np.isfinite(output[:, [0, -1]]).all()

