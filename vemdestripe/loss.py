"""Training objective used by vEMDestripe.

The objective is ``L1 + 0.1 Ledge + 0.3 Lphoto + 0.1 LSSIM``. The
photometric term matches per-image mean and standard deviation globally and
in non-overlapping windows of 32, 64, and 128 pixels. All inputs are grayscale
BCHW tensors in [0, 1].
"""

from __future__ import annotations

from collections.abc import Sequence

import torch
import torch.nn as nn
import torch.nn.functional as F


class VEMDestripeLoss(nn.Module):
    """Pixel, edge, photometric-statistics, and SSIM objective."""

    def __init__(
        self,
        lambda_l1: float = 1.0,
        lambda_edge: float = 0.1,
        lambda_photo: float = 0.3,
        lambda_ssim: float = 0.1,
        photo_std_weight: float = 0.5,
        local_window_sizes: Sequence[int] = (32, 64, 128),
        eps: float = 1e-6,
    ) -> None:
        super().__init__()
        if not local_window_sizes or any(int(size) <= 0 for size in local_window_sizes):
            raise ValueError("local_window_sizes must contain positive integers")

        self.mode = "D"
        self.lambda_l1 = float(lambda_l1)
        self.lambda_edge = float(lambda_edge)
        self.lambda_photo = float(lambda_photo)
        self.lambda_ssim = float(lambda_ssim)
        self.photo_std_weight = float(photo_std_weight)
        self.local_window_sizes = tuple(int(size) for size in local_window_sizes)
        self.eps = float(eps)

        sobel_x = torch.tensor(
            [[-1.0, 0.0, 1.0], [-2.0, 0.0, 2.0], [-1.0, 0.0, 1.0]]
        ).view(1, 1, 3, 3)
        sobel_y = torch.tensor(
            [[-1.0, -2.0, -1.0], [0.0, 0.0, 0.0], [1.0, 2.0, 1.0]]
        ).view(1, 1, 3, 3)
        self.register_buffer("sobel_x", sobel_x)
        self.register_buffer("sobel_y", sobel_y)

    @staticmethod
    def _check_inputs(pred: torch.Tensor, target: torch.Tensor) -> None:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target shapes differ: {pred.shape} vs {target.shape}")
        if pred.ndim != 4:
            raise ValueError(f"expected BCHW tensors, got {pred.ndim} dimensions")

    def edge_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Match Sobel gradient magnitudes, preserving the existing baseline term."""
        pred_gx = F.conv2d(pred, self.sobel_x, padding=1)
        pred_gy = F.conv2d(pred, self.sobel_y, padding=1)
        target_gx = F.conv2d(target, self.sobel_x, padding=1)
        target_gy = F.conv2d(target, self.sobel_y, padding=1)
        pred_grad = torch.sqrt(pred_gx.square() + pred_gy.square() + self.eps)
        target_grad = torch.sqrt(target_gx.square() + target_gy.square() + self.eps)
        return F.l1_loss(pred_grad, target_grad)

    def global_photometric_terms(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare mean and contrast independently for every image/channel."""
        dims = (-2, -1)
        pred_mean = pred.mean(dim=dims)
        target_mean = target.mean(dim=dims)
        pred_std = torch.sqrt(pred.var(dim=dims, unbiased=False) + self.eps)
        target_std = torch.sqrt(target.var(dim=dims, unbiased=False) + self.eps)
        return F.l1_loss(pred_mean, target_mean), F.l1_loss(pred_std, target_std)

    def local_photometric_terms(
        self, pred: torch.Tensor, target: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Compare local mean and contrast at several non-overlapping scales."""
        height, width = pred.shape[-2:]
        mean_terms: list[torch.Tensor] = []
        std_terms: list[torch.Tensor] = []

        for requested_size in self.local_window_sizes:
            size = min(requested_size, height, width)
            pred_mean = F.avg_pool2d(
                pred, kernel_size=size, stride=size, ceil_mode=True, count_include_pad=False
            )
            target_mean = F.avg_pool2d(
                target, kernel_size=size, stride=size, ceil_mode=True, count_include_pad=False
            )
            pred_mean_sq = F.avg_pool2d(
                pred.square(), kernel_size=size, stride=size, ceil_mode=True, count_include_pad=False
            )
            target_mean_sq = F.avg_pool2d(
                target.square(), kernel_size=size, stride=size, ceil_mode=True, count_include_pad=False
            )
            pred_std = torch.sqrt((pred_mean_sq - pred_mean.square()).clamp_min(0.0) + self.eps)
            target_std = torch.sqrt(
                (target_mean_sq - target_mean.square()).clamp_min(0.0) + self.eps
            )
            mean_terms.append(F.l1_loss(pred_mean, target_mean))
            std_terms.append(F.l1_loss(pred_std, target_std))

        return torch.stack(mean_terms).mean(), torch.stack(std_terms).mean()

    @staticmethod
    def _gaussian_mean(
        image: torch.Tensor,
        kernel_size: int,
        sigma: float = 1.5,
    ) -> torch.Tensor:
        """Gaussian local mean used by the standard SSIM formulation."""
        pad = kernel_size // 2
        padded = F.pad(image, (pad, pad, pad, pad), mode="reflect")
        coordinates = torch.arange(
            kernel_size,
            dtype=image.dtype,
            device=image.device,
        ) - float(pad)
        kernel_1d = torch.exp(-(coordinates.square()) / (2.0 * sigma**2))
        kernel_1d = kernel_1d / kernel_1d.sum()
        kernel_2d = torch.outer(kernel_1d, kernel_1d)
        kernel = kernel_2d.view(1, 1, kernel_size, kernel_size).expand(
            image.shape[1], 1, kernel_size, kernel_size
        )
        return F.conv2d(padded, kernel, groups=image.shape[1])

    def ssim_loss(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        """Differentiable single-scale SSIM loss for data in [0, 1]."""
        kernel_size = min(11, pred.shape[-2], pred.shape[-1])
        if kernel_size % 2 == 0:
            kernel_size -= 1
        if kernel_size < 3:
            return F.l1_loss(pred, target)

        mu_pred = self._gaussian_mean(pred, kernel_size)
        mu_target = self._gaussian_mean(target, kernel_size)
        var_pred = self._gaussian_mean(pred.square(), kernel_size) - mu_pred.square()
        var_target = self._gaussian_mean(target.square(), kernel_size) - mu_target.square()
        covariance = self._gaussian_mean(pred * target, kernel_size) - mu_pred * mu_target

        c1 = 0.01**2
        c2 = 0.03**2
        numerator = (2.0 * mu_pred * mu_target + c1) * (2.0 * covariance + c2)
        denominator = (mu_pred.square() + mu_target.square() + c1) * (
            var_pred.clamp_min(0.0) + var_target.clamp_min(0.0) + c2
        )
        mean_ssim = (numerator / denominator.clamp_min(self.eps)).mean().clamp(0.0, 1.0)
        return 1.0 - mean_ssim

    def forward(
        self,
        pred: torch.Tensor,
        target: torch.Tensor,
        ds_brightness: torch.Tensor | None = None,
        ds_contrast: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, dict[str, float]]:
        del ds_brightness, ds_contrast  # Kept for compatibility with the trainer call.
        self._check_inputs(pred, target)

        # Moment and variance estimates are sensitive to FP16 rounding. Keep
        # the loss calculation in FP32 even when the model forward uses AMP.
        pred_loss = pred.float()
        target_loss = target.float()

        l1 = F.l1_loss(pred_loss, target_loss)
        edge = (
            self.edge_loss(pred_loss, target_loss)
            if self.lambda_edge > 0
            else pred_loss.new_zeros(())
        )
        global_mean, global_std = self.global_photometric_terms(pred_loss, target_loss)
        global_photo = global_mean + self.photo_std_weight * global_std

        local_mean = pred_loss.new_zeros(())
        local_std = pred_loss.new_zeros(())
        local_mean, local_std = self.local_photometric_terms(pred_loss, target_loss)
        local_photo = local_mean + self.photo_std_weight * local_std
        photo = global_photo + local_photo

        ssim = self.ssim_loss(pred_loss, target_loss)
        total = (
            self.lambda_l1 * l1
            + self.lambda_edge * edge
            + self.lambda_photo * photo
            + self.lambda_ssim * ssim
        )

        diagnostics = {
            "l1": l1.item(),
            "edge": edge.item(),
            "brightness": photo.item(),
            "global_mean": global_mean.item(),
            "global_std": global_std.item(),
            "local_mean": local_mean.item(),
            "local_std": local_std.item(),
            "ssim": ssim.item(),
        }
        return total, diagnostics
