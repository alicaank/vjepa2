"""Latent PatchGAN helpers for V-JEPA token maps.

The discriminator operates directly on frozen-encoder latent token maps:
``(B, HW, C) -> (B, 1, H, W)`` patch logits. Loss helpers take a token mask
so callers can restrict the adversarial signal to true reveal/disocclusion
tokens while leaving deterministic transport regions to the L1 objective.
"""

from __future__ import annotations

import math
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm as apply_spectral_norm


def _maybe_spectral_norm(module: nn.Module, enabled: bool) -> nn.Module:
    return apply_spectral_norm(module) if enabled else module


def _infer_grid(hw: int, grid_size: Optional[Tuple[int, int]]) -> Tuple[int, int]:
    if grid_size is not None:
        gh, gw = int(grid_size[0]), int(grid_size[1])
        if gh <= 0 or gw <= 0 or gh * gw != hw:
            raise ValueError(
                f"grid_size={grid_size!r} is incompatible with HW={hw}"
            )
        return gh, gw
    side = int(math.sqrt(hw))
    if side * side != hw:
        raise ValueError(
            "LatentPatchDiscriminator needs grid_size for non-square token maps; "
            f"got HW={hw}."
        )
    return side, side


class LatentPatchDiscriminator(nn.Module):
    """Lightweight PatchGAN over flattened V-JEPA feature tokens.

    Args:
        in_dim: channel dimension of the concatenated hierarchical latent.
        hidden_dim: internal convolution width after the 1x1 projection.
        n_layers: number of spatial 3x3 refinement blocks.
        input_layer_norm: apply per-token LayerNorm before the convolutional
            stack. Defaults off so raw feature norms remain visible to the
            discriminator.
        spectral_norm: wrap conv layers with spectral norm for adversarial
            stability.
    """

    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 256,
        n_layers: int = 3,
        input_layer_norm: bool = False,
        spectral_norm: bool = True,
        leaky_relu_slope: float = 0.2,
    ) -> None:
        super().__init__()
        if in_dim <= 0:
            raise ValueError(f"in_dim must be positive, got {in_dim}")
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be positive, got {hidden_dim}")
        if n_layers < 0:
            raise ValueError(f"n_layers must be non-negative, got {n_layers}")

        self.in_dim = int(in_dim)
        self.hidden_dim = int(hidden_dim)
        self.input_norm = (
            nn.LayerNorm(self.in_dim) if input_layer_norm else nn.Identity()
        )

        layers: list[nn.Module] = [
            _maybe_spectral_norm(
                nn.Conv2d(self.in_dim, self.hidden_dim, kernel_size=1),
                spectral_norm,
            ),
            nn.LeakyReLU(leaky_relu_slope, inplace=True),
        ]
        for _ in range(int(n_layers)):
            layers.extend(
                [
                    _maybe_spectral_norm(
                        nn.Conv2d(
                            self.hidden_dim,
                            self.hidden_dim,
                            kernel_size=3,
                            padding=1,
                        ),
                        spectral_norm,
                    ),
                    nn.LeakyReLU(leaky_relu_slope, inplace=True),
                ]
            )
        layers.append(
            _maybe_spectral_norm(
                nn.Conv2d(self.hidden_dim, 1, kernel_size=1),
                spectral_norm,
            )
        )
        self.net = nn.Sequential(*layers)

    def forward(
        self,
        tokens: torch.Tensor,
        grid_size: Optional[Tuple[int, int]] = None,
    ) -> torch.Tensor:
        if tokens.ndim != 3:
            raise ValueError(
                "LatentPatchDiscriminator expects tokens with shape (B, HW, C); "
                f"got {tuple(tokens.shape)}."
            )
        B, HW, C = tokens.shape
        if C != self.in_dim:
            raise ValueError(f"Expected token dim {self.in_dim}, got {C}.")
        gh, gw = _infer_grid(HW, grid_size)
        x = self.input_norm(tokens)
        x = x.transpose(1, 2).reshape(B, C, gh, gw)
        return self.net(x)


def mask_to_patch_logits(mask: torch.Tensor, logits: torch.Tensor) -> torch.Tensor:
    """Reshape a token mask to ``(B, 1, H, W)`` for PatchGAN logits."""
    if logits.ndim != 4 or logits.shape[1] != 1:
        raise ValueError(
            f"logits must have shape (B, 1, H, W); got {tuple(logits.shape)}"
        )
    B, _, H, W = logits.shape
    if mask.ndim == 3 and mask.shape[-1] == 1:
        mask = mask.squeeze(-1)
    if mask.ndim == 4 and mask.shape[1] == 1:
        mask = mask[:, 0]
    if mask.ndim != 2:
        raise ValueError(
            "mask must have shape (B, HW), (B, HW, 1), or (B, 1, H, W); "
            f"got {tuple(mask.shape)}"
        )
    if mask.shape[0] != B or mask.shape[1] != H * W:
        raise ValueError(
            f"mask shape {tuple(mask.shape)} is incompatible with logits "
            f"shape {tuple(logits.shape)}"
        )
    return mask.to(device=logits.device, dtype=logits.dtype).reshape(B, 1, H, W)


def masked_patch_mean(
    values: torch.Tensor,
    mask: torch.Tensor,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Mean over selected patch logits, returning differentiable zero if empty."""
    patch_mask = mask_to_patch_logits(mask, values)
    denom = patch_mask.sum()
    masked_sum = (values * patch_mask).sum()
    return torch.where(
        denom > 0,
        masked_sum / denom.clamp_min(eps),
        masked_sum * 0.0,
    )


def latent_patchgan_hinge_discriminator_loss(
    real_logits: torch.Tensor,
    fake_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked hinge loss for the discriminator."""
    if real_logits.shape != fake_logits.shape:
        raise ValueError(
            f"real/fake logits must share shape; got {tuple(real_logits.shape)} "
            f"and {tuple(fake_logits.shape)}"
        )
    real_loss = masked_patch_mean(F.relu(1.0 - real_logits), mask)
    fake_loss = masked_patch_mean(F.relu(1.0 + fake_logits), mask)
    return real_loss + fake_loss


def latent_patchgan_hinge_generator_loss(
    fake_logits: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    """Masked hinge generator loss: maximize discriminator realism score."""
    return masked_patch_mean(-fake_logits, mask)


__all__ = [
    "LatentPatchDiscriminator",
    "latent_patchgan_hinge_discriminator_loss",
    "latent_patchgan_hinge_generator_loss",
    "masked_patch_mean",
    "mask_to_patch_logits",
]
