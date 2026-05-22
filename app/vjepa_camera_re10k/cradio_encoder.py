"""C-RADIO encoder adapter for the camera-AC predictor.

This is the same integration shape as ``mast3r_encoder.py``: keep the
canonical camera predictor/training loop intact, but swap the frozen feature
stream. C-RADIO exposes a single spatial feature map rather than V-JEPA's
four tapped transformer blocks, so the expected predictor contract is
``n_hierarchical_layers=1``.
"""

from __future__ import annotations

import logging
from typing import List

import torch
from torch import nn
from torch.nn import functional as F

logger = logging.getLogger(__name__)

DEFAULT_CRADIO_MODEL_VERSION = "c-radio_v4-h"
DEFAULT_FEATURE_DIMS = {
    "c-radio_v4-h": 1280,
    "c-radio_v4-so400m": 1152,
}


def _bool_env(value: str | None, default: bool = False) -> bool:
    if value is None:
        return default
    return value.lower() in ("1", "true", "yes", "y", "on")


class CRADIOEncoderAdapter(nn.Module):
    """Frozen C-RADIO spatial-feature adapter.

    Input shapes:
        ``(B, 3, T, H, W)`` or ``(B, 3, H, W)`` with image values in the
        dataset's normal camera pipeline range. If values look like ImageNet
        normalized tensors, they are approximately inverted before calling
        RADIO, which expects ``[0, 1]`` RGB.

    Output:
        A one-element list containing ``(B, T*N, D)`` feature tokens.
    """

    def __init__(
        self,
        *,
        model_version: str = DEFAULT_CRADIO_MODEL_VERSION,
        img_size: int = 384,
        patch_size: int = 16,
        embed_dim: int | None = None,
        freeze: bool = True,
        force_reload: bool = False,
        skip_validation: bool = True,
    ) -> None:
        super().__init__()
        self._frozen: bool = False
        self.model_version = str(model_version)
        self.img_size = int(img_size)
        self.patch_size = int(patch_size)
        self.tokens_per_image = (self.img_size // self.patch_size) ** 2
        self.img_temporal_dim_size = 1
        self.hierarchical_layers = [0]
        self.out_layers = [0]
        self.num_heads = 16

        if self.img_size % self.patch_size != 0:
            raise ValueError(
                f"img_size={self.img_size} not divisible by patch_size={self.patch_size}"
            )

        logger.info(
            "Loading C-RADIO encoder via torch.hub: "
            f"version={self.model_version}, force_reload={force_reload}"
        )
        self.model = torch.hub.load(
            "NVlabs/RADIO",
            "radio_model",
            version=self.model_version,
            progress=True,
            skip_validation=skip_validation,
            force_reload=force_reload,
        )

        inferred_dim = self._infer_feature_dim(self.model)
        if embed_dim is None:
            embed_dim = inferred_dim
        self.embed_dim = int(embed_dim)

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            self.eval()
            self._frozen = True

        logger.info(
            "CRADIOEncoderAdapter ready: "
            f"version={self.model_version}, img_size={self.img_size}, "
            f"patch_size={self.patch_size}, embed_dim={self.embed_dim}, "
            f"hierarchical_layers={self.hierarchical_layers}, frozen={self._frozen}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    @staticmethod
    def _infer_feature_dim(model: nn.Module) -> int:
        for name in (
            "embed_dim",
            "width",
            "num_features",
            "feature_dim",
            "output_dim",
        ):
            value = getattr(model, name, None)
            if isinstance(value, int) and value > 0:
                return int(value)
        for child_name in ("model", "backbone", "vision_model", "trunk"):
            child = getattr(model, child_name, None)
            if child is not None:
                for name in (
                    "embed_dim",
                    "width",
                    "num_features",
                    "feature_dim",
                    "output_dim",
                ):
                    value = getattr(child, name, None)
                    if isinstance(value, int) and value > 0:
                        return int(value)
        version = getattr(model, "version", None)
        if isinstance(version, str) and version in DEFAULT_FEATURE_DIMS:
            return DEFAULT_FEATURE_DIMS[version]
        return DEFAULT_FEATURE_DIMS[DEFAULT_CRADIO_MODEL_VERSION]

    @staticmethod
    def _to_radio_rgb01(x: torch.Tensor) -> torch.Tensor:
        # Most camera runs feed ImageNet-normalized tensors. Convert those
        # back to RGB [0, 1] for RADIO. If the tensor is already [0, 1], this
        # branch leaves it untouched.
        if float(x.detach().amin()) < -0.05 or float(x.detach().amax()) > 1.05:
            mean = x.new_tensor([0.485, 0.456, 0.406]).view(1, 3, 1, 1)
            std = x.new_tensor([0.229, 0.224, 0.225]).view(1, 3, 1, 1)
            x = x * std + mean
        return x.clamp_(0.0, 1.0)

    @torch.no_grad()
    def _maybe_no_grad_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._encode(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        if self._frozen:
            return self._maybe_no_grad_forward(x)
        return self._encode(x)

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        elif x.ndim == 4:
            B, C, H, W = x.shape
            T = 1
        else:
            raise ValueError(
                f"CRADIOEncoderAdapter expected 4D or 5D input; got shape {tuple(x.shape)}"
            )
        if C != 3:
            raise ValueError(f"CRADIOEncoderAdapter expected 3 channels; got {C}")

        x = self._to_radio_rgb01(x)
        nearest_res = self.model.get_nearest_supported_resolution(H, W)
        if tuple(nearest_res) != (H, W):
            x = F.interpolate(x, nearest_res, mode="bilinear", align_corners=False)

        autocast_enabled = x.is_cuda
        with torch.autocast("cuda", dtype=torch.bfloat16, enabled=autocast_enabled):
            output = self.model(x, feature_fmt="NCHW")

        if isinstance(output, dict):
            output = output.get("backbone", next(iter(output.values())))
        _, spatial = output
        if spatial.ndim == 3:
            n = spatial.shape[1]
            side = int(n ** 0.5)
            if side * side != n:
                raise RuntimeError(
                    f"C-RADIO returned NLC features with non-square token count {n}"
                )
            spatial = spatial.transpose(1, 2).reshape(spatial.shape[0], spatial.shape[2], side, side)
        elif spatial.ndim != 4:
            raise RuntimeError(
                f"C-RADIO returned unexpected spatial feature shape {tuple(spatial.shape)}"
            )

        target_side = self.img_size // self.patch_size
        if spatial.shape[-2:] != (target_side, target_side):
            spatial = F.interpolate(
                spatial.float(),
                size=(target_side, target_side),
                mode="bilinear",
                align_corners=False,
            ).to(dtype=spatial.dtype)

        tokens = spatial.flatten(2).transpose(1, 2).contiguous()
        if tokens.shape[-1] != self.embed_dim:
            raise RuntimeError(
                f"C-RADIO feature dim mismatch: adapter configured embed_dim={self.embed_dim} "
                f"but forward returned {tokens.shape[-1]}. Set CRADIO_EMBED_DIM to match."
            )
        if T > 1:
            tokens = tokens.view(B, T, tokens.shape[1], tokens.shape[2]).reshape(
                B, T * tokens.shape[1], tokens.shape[2]
            )
        return [tokens]
