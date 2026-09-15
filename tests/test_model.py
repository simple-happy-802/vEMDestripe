import torch

from vemdestripe.model import ChannelSpatialGate, VEMDestripe


def test_parameter_count_and_output_shape():
    model = VEMDestripe()
    assert sum(parameter.numel() for parameter in model.parameters()) == 1_909_811
    with torch.no_grad():
        output = model(torch.rand(1, 2, 64, 64))
    assert output.shape == (1, 1, 64, 64)


def test_csg_is_identity_at_initialization():
    gate = ChannelSpatialGate(32)
    feature = torch.randn(2, 32, 11, 13)
    with torch.no_grad():
        output = gate(feature)
    torch.testing.assert_close(output, feature, rtol=0, atol=0)

