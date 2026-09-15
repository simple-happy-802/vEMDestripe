from pathlib import Path

import torch

from vemdestripe.model import VEMDestripe


def test_included_checkpoint_matches_public_model():
    path = Path(__file__).resolve().parents[1] / "checkpoints" / "vemdestripe_seed42" / "best.pth"
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    assert checkpoint["method_name"] == "vEMDestripe"
    assert checkpoint["reference_preprocess"] == "block_mean"
    assert checkpoint["reference_block_size"] == 16
    assert checkpoint["num_reference"] == 16
    result = VEMDestripe().load_state_dict(checkpoint["model_state_dict"], strict=True)
    assert not result.missing_keys
    assert not result.unexpected_keys

