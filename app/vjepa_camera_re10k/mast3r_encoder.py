"""MASt3R encoder adapter for the camera-AC predictor.

Wraps the encoder side of ``AsymmetricMASt3R`` so it exposes the same
image-path forward contract as V-JEPA 2.1 ViT-L:

* accepts ``(B, C, 1, H, W)`` clips (single-frame "image-path") OR
  ``(B, C, H, W)`` plain image batches,
* returns a ``list[Tensor]`` of per-layer activations, each shape
  ``(B, N, D)`` with ``N = (H/patch)^2`` and ``D = 1024``,
* indexes layers from ``hierarchical_layers`` (canonical
  ``[5, 11, 17, 23]`` for the 24-block CroCo ViT-L backbone).

Compatibility surface required by ``init_video_model`` /
``encode_clip_as_images`` (see ``utils.py``):

    * ``embed_dim`` int
    * ``num_heads`` int
    * ``hierarchical_layers`` list[int]
    * ``out_layers`` list[int]
    * ``img_temporal_dim_size`` int (``1`` -> image path)
    * ``forward(x)`` returning a *list* of per-layer tensors.

The adapter is intentionally minimal: no temporal layers, no per-tap
RoPE re-derivation, no distillation hooks. Where the predictor's
upstream V-JEPA 2.1 forward returns hierarchically-LayerNorm'd features
via ``norms_block``, we mirror the same contract by cloning the
encoder's final ``enc_norm`` once per requested tap. The clones are
*frozen* by default (matching the user-selected ``enc_lr_scale=0.0``
protocol for the MASt3R pilot) but are still real ``nn.LayerNorm``
modules so a future fine-tune sweep can simply un-freeze them.

This file lives under ``thirdparty/vjepa2/app/vjepa_camera_re10k/``
because (a) it is consumed exclusively by ``utils.py:init_video_model``
in that package, and (b) it must not introduce a hard dependency on
``mast3r`` for users running the canonical V-JEPA 2.1 pipeline. Imports
from ``mast3r.model`` therefore live *inside* the adapter constructor.
"""

from __future__ import annotations

import copy
import logging
from typing import List, Sequence

import torch
from torch import nn

logger = logging.getLogger(__name__)

# Canonical MASt3R ViT-L checkpoint. The same string is used by the
# multiscene Gaussian-decoder pipeline (see
# ``src/training/multiscene/trainer.py``), so a single ``HF_HOME`` cache
# is shared across both training entrypoints.
DEFAULT_MAST3R_CHECKPOINT = (
    "naver/MASt3R_ViTLarge_BaseDecoder_512_catmlpdpt_metric"
)

# CroCo ViT-Large = 24 encoder blocks, embed_dim=1024, 16 heads,
# patch_size=16. The hierarchical taps below match V-JEPA 2.1 ViT-L's
# canonical ``[5, 11, 17, 23]`` so the predictor does not see a layer-
# index distribution shift when the backbone is swapped.
DEFAULT_HIERARCHICAL_LAYERS: tuple[int, ...] = (5, 11, 17, 23)


class MASt3REncoderAdapter(nn.Module):
    """Encoder-side wrapper around a pretrained ``AsymmetricMASt3R``.

    Args:
        checkpoint_id: HuggingFace repo id for the MASt3R checkpoint.
            Defaults to the ViT-L 512 metric variant used elsewhere in
            this codebase.
        img_size: spatial input size in pixels. MASt3R was trained at
            512 but is a fully-convolutional ViT and accepts any
            multiple of ``patch_size``. We default to 384 so the token
            grid (24x24 = 576) matches the V-JEPA 2.1 ViT-L pipeline
            byte-for-byte; no downstream geometry tables need to change.
        patch_size: must divide ``img_size``. Default 16.
        out_layers: list of block indices to tap. Must be a subset of
            ``hierarchical_layers``. Defaults to ``hierarchical_layers``.
        hierarchical_layers: layer indices considered "canonical" for
            this encoder family. The predictor's per-layer projection
            heads are sized by ``len(hierarchical_layers)``.
        freeze: whether to set ``requires_grad=False`` on all encoder
            parameters (including the cloned per-tap LayerNorms). The
            MASt3R pilot uses ``True`` (matching the user-selected
            ``enc_lr_scale=0.0`` protocol). Set ``False`` for a future
            light fine-tune ablation.

    Forward:
        Input ``x`` shape: ``(B, 3, 1, H, W)`` (V-JEPA image path) or
        ``(B, 3, H, W)`` (plain image batch). The temporal dim is
        squeezed.
        Output: list of ``len(out_layers)`` tensors, each
        ``(B, N, embed_dim)`` with ``N = (H/patch_size)**2``.
    """

    def __init__(
        self,
        *,
        checkpoint_id: str = DEFAULT_MAST3R_CHECKPOINT,
        img_size: int = 384,
        patch_size: int = 16,
        out_layers: Sequence[int] | None = None,
        hierarchical_layers: Sequence[int] | None = None,
        freeze: bool = True,
    ) -> None:
        super().__init__()
        # Defensive: the overridden ``train()`` reads ``self._frozen``,
        # so it must exist before any code path that could trigger it
        # (e.g. ``self.eval()`` below).
        self._frozen: bool = False

        if img_size % patch_size != 0:
            raise ValueError(
                f"img_size={img_size} not divisible by patch_size={patch_size}"
            )

        # Lazy import so users running the V-JEPA 2.1 pipeline do not
        # need ``mast3r`` installed.
        try:
            from mast3r.model import AsymmetricMASt3R
        except ImportError as exc:
            raise ImportError(
                "MASt3REncoderAdapter requires the `mast3r` package. "
                "Install MASt3R from https://github.com/naver/mast3r "
                "or pip-install via the project's environment."
            ) from exc

        logger.info(f"Loading MASt3R encoder from {checkpoint_id} ...")
        full_model = AsymmetricMASt3R.from_pretrained(checkpoint_id)
        if full_model.enc_pos_embed is not None:
            # Defensive: MASt3R relies on RoPE inside each block, with
            # ``enc_pos_embed`` deliberately None in the canonical
            # checkpoints. Falling back to additive pos-embeds would
            # introduce a silent semantic shift.
            raise RuntimeError(
                "MASt3R encoder unexpectedly has an additive enc_pos_embed; "
                "this adapter only supports the canonical RoPE-only checkpoint."
            )

        # Resolve canonical layer set.
        if hierarchical_layers is None:
            n_blocks = len(full_model.enc_blocks)
            if n_blocks == 24:
                hierarchical_layers = DEFAULT_HIERARCHICAL_LAYERS
            else:
                # Generic 4-tap policy mirroring V-JEPA's
                # ``depth // {4, 2, 4/3, 1} - 1``.
                hierarchical_layers = (
                    n_blocks // 4 - 1,
                    n_blocks // 2 - 1,
                    3 * n_blocks // 4 - 1,
                    n_blocks - 1,
                )
        hierarchical_layers = tuple(int(i) for i in hierarchical_layers)

        if out_layers is None:
            out_layers = hierarchical_layers
        out_layers = tuple(int(i) for i in out_layers)

        # Each ``out_layers`` entry must be tappable inside the encoder.
        for li in out_layers:
            if li < 0 or li >= len(full_model.enc_blocks):
                raise ValueError(
                    f"out_layers index {li} out of range "
                    f"[0, {len(full_model.enc_blocks)})."
                )
        # The predictor's per-layer head split is keyed off
        # ``hierarchical_layers``; out_layers must therefore be a subset.
        for li in out_layers:
            if li not in hierarchical_layers:
                raise ValueError(
                    f"out_layers={list(out_layers)} contains {li} which is not in "
                    f"hierarchical_layers={list(hierarchical_layers)}. Only "
                    f"layers in hierarchical_layers carry a per-tap LayerNorm."
                )

        # Pull only the encoder side of MASt3R into the adapter. The
        # asymmetric decoder (cross-attention pointmap predictor) is
        # never invoked in our pipeline; dropping it here keeps the
        # parameter count and DDP broadcast surface minimal.
        self.patch_embed = full_model.patch_embed
        self.enc_blocks = full_model.enc_blocks
        self.rope = full_model.rope  # held by each block; kept here for completeness
        self.enc_norm = full_model.enc_norm
        # Per-tap LayerNorms cloned from ``enc_norm``. V-JEPA 2.1's
        # forward applies ``norms_block[i]`` to the i-th tap; we mirror
        # that contract so the predictor sees comparably-scaled features
        # at every tap. Cloned (not shared) so a future fine-tune can
        # decouple them without touching this file.
        self.norms_block = nn.ModuleList(
            [copy.deepcopy(full_model.enc_norm) for _ in out_layers]
        )

        # Public, V-JEPA-compatible attributes consumed by the rest of
        # the pipeline.
        self.embed_dim: int = int(full_model.enc_embed_dim)
        self.num_heads: int = int(full_model.enc_blocks[0].attn.num_heads)
        self.img_size: int = int(img_size)
        self.patch_size: int = int(patch_size)
        self.tokens_per_image: int = (img_size // patch_size) ** 2
        # ``img_temporal_dim_size=1`` mirrors V-JEPA 2.1's image-path flag.
        # ``encode_clip_as_images`` checks this before reshaping.
        self.img_temporal_dim_size: int = 1
        self.hierarchical_layers: list[int] = list(hierarchical_layers)
        self.out_layers: list[int] = list(out_layers)
        self.out_layer_set: set[int] = set(self.out_layers)

        # Drop reference to the original full model so the asymmetric
        # decoder is GC'd.
        del full_model

        if freeze:
            for p in self.parameters():
                p.requires_grad_(False)
            # ``eval()`` so dropout / batch-stat behaviour matches the
            # frozen-encoder use case. The training loop calls
            # ``encoder.train()`` on every epoch start; we override
            # ``train()`` below so that does not undo this.
            self.eval()
            self._frozen = True
        else:
            self._frozen = False

        logger.info(
            f"MASt3REncoderAdapter ready: img_size={self.img_size}, "
            f"patch_size={self.patch_size}, embed_dim={self.embed_dim}, "
            f"num_heads={self.num_heads}, hierarchical_layers={self.hierarchical_layers}, "
            f"out_layers={self.out_layers}, frozen={self._frozen}, "
            f"params={sum(p.numel() for p in self.parameters()):,}"
        )

    def train(self, mode: bool = True):  # type: ignore[override]
        """Respect the frozen flag against the training loop's ``encoder.train()`` call.

        The V-JEPA pipeline does ``encoder.train()`` once per epoch; for
        a frozen MASt3R encoder we do not want dropout / batch-stat
        toggling. ``forward`` runs under ``torch.no_grad()`` for frozen
        encoders anyway, but keeping ``training=False`` is the
        defensive choice.
        """
        if self._frozen:
            return super().train(False)
        return super().train(mode)

    @torch.no_grad()
    def _maybe_no_grad_forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        return self._encode(x)

    def forward(self, x: torch.Tensor) -> List[torch.Tensor]:
        """Run MASt3R encoder, capturing per-tap activations.

        Input shapes:
            ``(B, 3, T, H, W)``  — multi-frame clip (V-JEPA's video path
                with ``tubelet_size=1``); each frame is encoded
                independently and the per-frame token grids are
                concatenated along the token axis to produce a flat
                ``(t, y, x)`` ordering identical to V-JEPA's video-path
                output. Cross-frame mixing is the predictor's job.
            ``(B, 3, 1, H, W)``  — single-frame degenerate case of the
                above.
            ``(B, 3, H, W)``     — single-frame image-path call.

        Output: list of ``len(out_layers)`` tensors, each
        ``(B, T * P * P, D)`` (or ``(B, P * P, D)`` for 4D input).
        """
        if self._frozen:
            return self._maybe_no_grad_forward(x)
        return self._encode(x)

    def _encode(self, x: torch.Tensor) -> List[torch.Tensor]:
        # Normalize to a 4D ``(B*T, 3, H, W)`` tensor for the per-frame
        # ViT, remembering ``T`` so we can reshape token grids back to
        # the V-JEPA per-frame ordering ``(B, T*N_per_frame, D)``.
        if x.ndim == 5:
            B, C, T, H, W = x.shape
            # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W) so the
            # outer order is "all frames of clip 0, then clip 1, ..."
            # matching V-JEPA Conv3d(kernel=1) which flattens (B, C, T, P, P)
            # into (B, T*P*P, C) in the same (b, t, y, x) order.
            x = x.permute(0, 2, 1, 3, 4).contiguous().view(B * T, C, H, W)
        elif x.ndim == 4:
            B, C, H, W = x.shape
            T = 1
        else:
            raise ValueError(
                f"MASt3REncoderAdapter expected 4D or 5D input; got shape "
                f"{tuple(x.shape)}"
            )

        if (H, W) != (self.img_size, self.img_size):
            raise ValueError(
                f"MASt3REncoderAdapter built for img_size={self.img_size} "
                f"but received {(H, W)}. Resize callers (e.g. the dataset "
                f"transform) instead of resizing here, so the rest of the "
                f"pipeline (ray PE, warp tables, etc.) sees a consistent "
                f"crop_size."
            )

        BT = x.shape[0]  # B * T

        # MASt3R's PatchEmbedDust3R requires ``true_shape`` per batch
        # element (used to compute the right RoPE coordinates when
        # images may be padded/cropped). For our pipeline every clip is
        # a fixed crop so we hand it the trivial all-equal tensor.
        true_shape = torch.tensor(
            [[H, W]], dtype=torch.int32, device=x.device
        ).expand(BT, 2).contiguous()

        feats, pos = self.patch_embed(x, true_shape=true_shape)  # feats: (B*T, N, D), pos: (B*T, N, 2)

        outputs: List[torch.Tensor] = [None] * len(self.out_layers)  # type: ignore[list-item]
        for blk_idx, blk in enumerate(self.enc_blocks):
            feats = blk(feats, pos)
            if blk_idx in self.out_layer_set:
                # First normalize (per-tap LayerNorm clone of enc_norm)
                # so the predictor sees comparably-scaled features at
                # every tap, matching V-JEPA's ``norms_block`` contract.
                slot = self.out_layers.index(blk_idx)
                normed = self.norms_block[slot](feats)  # (B*T, N_per_frame, D)
                if T > 1:
                    # Merge per-frame token grids into the V-JEPA flat
                    # ordering ``(b, t, y, x)``.
                    N_per_frame = normed.shape[1]
                    D = normed.shape[2]
                    normed = normed.view(B, T, N_per_frame, D).reshape(
                        B, T * N_per_frame, D
                    )
                outputs[slot] = normed

        # Sanity: every requested tap was filled.
        if any(o is None for o in outputs):
            missing = [
                self.out_layers[i] for i, o in enumerate(outputs) if o is None
            ]
            raise RuntimeError(
                f"MASt3REncoderAdapter forward did not capture taps {missing}; "
                f"this should be unreachable."
            )

        return outputs  # type: ignore[return-value]
