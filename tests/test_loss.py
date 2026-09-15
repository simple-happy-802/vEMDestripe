import torch

from vemdestripe.loss import VEMDestripeLoss


def test_loss_is_finite_and_differentiable():
    prediction = torch.rand(2, 1, 64, 64, requires_grad=True)
    target = torch.rand(2, 1, 64, 64)
    total, terms = VEMDestripeLoss()(prediction, target)
    assert torch.isfinite(total)
    assert {"l1", "edge", "global_mean", "global_std", "local_mean", "local_std", "ssim"} <= set(terms)
    total.backward()
    assert prediction.grad is not None
    assert torch.isfinite(prediction.grad).all()

