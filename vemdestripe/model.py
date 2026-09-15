"""Neural network used by vEMDestripe.

A stripe-affected XY
section and one structure-suppressed reference are concatenated as two input
channels. The network has three encoder/decoder levels, additive skip
connections, two bottleneck feature-transform blocks, and nine
channel-spatial gated (CSG) units.

Attribute names intentionally match the development checkpoint implementation. Its
``model_state_dict`` therefore loads strictly without renaming parameters.
"""

from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class ChannelSpatialGate(nn.Module):
    """Sequential channel/spatial gating with an identity initialization.

    The output is ``x + gamma * gated(x)`` and ``gamma`` starts at zero. Thus
    a CSG is an exact identity at initialization and cannot suppress features
    before its gated response has been learned.
    """

    def __init__(self, channels: int, reduction: int = 4) -> None:
        super().__init__()
        hidden = max(1, channels // reduction)
        self.mode = "full"
        self.residual_fix = True
        self.channel_attn = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(channels, hidden, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, channels, 1),
            nn.Sigmoid(),
        )
        self.spatial_attn = nn.Sequential(
            nn.Conv2d(channels, hidden, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(hidden, 1, 1),
            nn.Sigmoid(),
        )
        self.gamma = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        identity = x
        gated = x * self.channel_attn(x)
        gated = gated * self.spatial_attn(gated)
        return identity + self.gamma * gated


class FeatureTransformBlock(nn.Module):
    """CSG branch followed by a pointwise feed-forward branch."""

    def __init__(self, channels: int, reduction: int = 4, use_norm: bool = True) -> None:
        super().__init__()
        self.norm1 = nn.GroupNorm(8, channels) if use_norm else nn.Identity()
        self.attn = ChannelSpatialGate(channels, reduction=reduction)
        self.norm2 = nn.GroupNorm(8, channels) if use_norm else nn.Identity()
        self.ffn = nn.Sequential(
            nn.Conv2d(channels, channels * 2, 1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(channels * 2, channels, 1),
        )
        self.gamma1 = nn.Parameter(torch.zeros(1))
        self.gamma2 = nn.Parameter(torch.zeros(1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x + self.gamma1 * self.attn(self.norm1(x))
        x = x + self.gamma2 * self.ffn(self.norm2(x))
        return x


class VEMDestripe(nn.Module):
    """Reference-guided 2-D restoration network reported in the paper."""

    def __init__(
        self,
        in_channels: int = 2,
        out_channels: int = 1,
        base_features: int = 32,
        num_blocks: int = 2,
        **compatibility_kwargs,
    ) -> None:
        super().__init__()
        # Historical constructors supplied these fields. They are accepted
        # only to load old checkpoints; the public architecture is fixed.
        compatibility_kwargs.pop("use_z_context", None)
        compatibility_kwargs.pop("attention_mode", None)
        compatibility_kwargs.pop("architecture_ablation", None)
        compatibility_kwargs.pop("photo_field_size", None)
        if compatibility_kwargs:
            raise TypeError(f"Unsupported model arguments: {sorted(compatibility_kwargs)}")

        self.use_z_context = False
        self.attention_mode = "full"
        self.method_name = "vEMDestripe"
        self.out_channels = int(out_channels)

        # Sequential indices match the development checkpoints exactly.
        self.input_conv = nn.Sequential(
            nn.Conv2d(in_channels, base_features, 3, padding=1),
            nn.GroupNorm(8, base_features),
            nn.LeakyReLU(0.2, inplace=True),
            ChannelSpatialGate(base_features),
        )
        self.enc1 = self._make_encoder_block(base_features, base_features * 2)
        self.enc2 = self._make_encoder_block(base_features * 2, base_features * 4)
        self.enc3 = self._make_encoder_block(base_features * 4, base_features * 8)

        dim = base_features * 8
        self.transformer_blocks = nn.ModuleList(
            [FeatureTransformBlock(dim) for _ in range(num_blocks)]
        )
        self.dec3 = self._make_decoder_block(base_features * 8, base_features * 4)
        self.dec2 = self._make_decoder_block(base_features * 4, base_features * 2)
        self.dec1 = self._make_decoder_block(base_features * 2, base_features)
        self.output = nn.Sequential(
            nn.Conv2d(base_features, base_features, 3, padding=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(base_features, out_channels, 1),
        )

    @staticmethod
    def _make_encoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(in_channels, out_channels, 3, stride=2, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            FeatureTransformBlock(out_channels),
        )

    @staticmethod
    def _make_decoder_block(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            nn.ConvTranspose2d(in_channels, out_channels, 2, stride=2),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1),
            nn.GroupNorm(8, out_channels),
            nn.LeakyReLU(0.2, inplace=True),
            FeatureTransformBlock(out_channels),
        )

    def forward(self, x: torch.Tensor, z_context=None) -> torch.Tensor:
        del z_context
        x0 = self.input_conv(x)
        encoder_level1 = self.enc1(x0)
        encoder_level2 = self.enc2(encoder_level1)
        encoder_level3 = self.enc3(encoder_level2)

        features = encoder_level3
        for block in self.transformer_blocks:
            features = block(features)

        decoder_level3 = self.dec3(features)
        decoder_level3 = decoder_level3 + F.interpolate(
            encoder_level2,
            size=decoder_level3.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        decoder_level2 = self.dec2(decoder_level3)
        decoder_level2 = decoder_level2 + F.interpolate(
            encoder_level1,
            size=decoder_level2.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        decoder_level1 = self.dec1(decoder_level2)
        decoder_level1 = decoder_level1 + F.interpolate(
            x0,
            size=decoder_level1.shape[-2:],
            mode="bilinear",
            align_corners=True,
        )
        return self.output(decoder_level1)


def create_model() -> VEMDestripe:
    """Create the paper network configuration."""
    return VEMDestripe(in_channels=2, out_channels=1, base_features=32, num_blocks=2)
