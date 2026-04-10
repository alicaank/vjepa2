# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""
Camera-Conditioned V-JEPA 2.1 Predictor
========================================
Fuses:
  - The per-frame action/state/intrinsics interleaving + causal mask of VisionTransformerPredictorAC
    (src/models/ac_predictor.py), using the ACBlock which implements action-token-aware RoPE attention.
  - The multi-layer hierarchical input embedding + dense predictive loss output heads of
    VisionTransformerPredictor from vjepa_2_1/models/predictor.py.

Key fixes over the naive merge proposed externally:
  1. Uses ACBlock (not a plain Block) so action tokens get correct RoPE treatment.
  2. Conditioning tokens are interleaved PER FRAME as [action, state, (intrinsics), patch_0..patch_HW-1]
     matching the causal-mask layout, NOT prepended as a flat prefix.
  3. predictor_embed compresses multi-layer concatenated encoder features (L * embed_dim) → predictor_embed_dim
     via a small MLP when n_hierarchical_layers > 1 (matching 2.1 predictor).
  4. intrinsics_dim is a free parameter (no hardcoded action_embed_dim - 1 like DROID).
  5. Dense context head (predictor_proj_context) mirrors the 2.1 predictor when predict_all=True.
"""

import math
from functools import partial

import torch
import torch.nn as nn

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_


class CameraConditionedPredictorAC(nn.Module):
    """
    Camera-pose-conditioned predictor with V-JEPA 2.1 hierarchical output heads.

    Args:
        img_size: spatial resolution of encoder input (H, W) or int.
        patch_size: patch size used by the encoder.
        num_frames: number of input video frames per clip.
        tubelet_size: temporal tubelet stride of the 3-D tokenizer.
        embed_dim: encoder output dimension per token (single layer).
        predictor_embed_dim: internal predictor width.
        n_hierarchical_layers: number of encoder layers whose features are
            concatenated as input.  If > 1, embed_dim * n_hierarchical_layers
            is compressed to predictor_embed_dim by a small MLP.
        out_embed_dim: encoder output dim used for loss targets.  If None,
            defaults to embed_dim.  predictor_proj outputs
            n_hierarchical_layers * out_embed_dim features so the loss can
            compare each predicted layer independently.
        depth: number of transformer blocks.
        num_heads: attention heads.
        mlp_ratio: MLP expansion ratio.
        state_dim: dimensionality of absolute camera pose token (default 7:
            3D translation + 4D quaternion).
        action_dim: dimensionality of relative camera motion token (default 7).
        intrinsics_dim: dimensionality of camera intrinsics token (default 4:
            normalized fx, fy, cx, cy).  Set to 0 to disable.
        use_intrinsics: whether to include an intrinsics conditioning token.
        predict_all: if True, also project visible context tokens for dense
            loss (V-JEPA 2.1 style).
        is_frame_causal: whether to apply block-causal attention masking.
        use_rope: use Rotary Position Embedding in attention.
        use_silu: use SiLU / SwiGLU instead of GELU in MLP.
        wide_silu: use the wide SwiGLU variant.
        use_activation_checkpointing: gradient checkpointing for memory saving.
    """

    def __init__(
        self,
        img_size=(224, 224),
        patch_size=16,
        num_frames=16,
        tubelet_size=2,
        embed_dim=1024,
        predictor_embed_dim=1024,
        n_hierarchical_layers=4,
        out_embed_dim=None,
        depth=12,
        num_heads=16,
        mlp_ratio=4.0,
        qkv_bias=True,
        qk_scale=None,
        drop_rate=0.0,
        attn_drop_rate=0.0,
        drop_path_rate=0.0,
        norm_layer=nn.LayerNorm,
        init_std=0.02,
        uniform_power=True,
        state_dim=7,
        action_dim=7,
        intrinsics_dim=4,
        use_intrinsics=True,
        predict_all=True,
        is_frame_causal=True,
        use_rope=True,
        use_silu=False,
        wide_silu=True,
        use_activation_checkpointing=False,
        **kwargs,
    ):
        super().__init__()

        self.is_frame_causal = is_frame_causal
        self.use_intrinsics = use_intrinsics and (intrinsics_dim > 0)
        self.predict_all = predict_all
        self.n_hierarchical_layers = n_hierarchical_layers
        self.use_activation_checkpointing = use_activation_checkpointing
        self.init_std = init_std

        # ------------------------------------------------------------------ #
        # Grid geometry (mirrors VisionTransformerPredictorAC)
        # ------------------------------------------------------------------ #
        if isinstance(img_size, int):
            img_size = (img_size, img_size)
        self.img_height, self.img_width = img_size
        self.patch_size = patch_size
        self.num_frames = num_frames
        self.tubelet_size = tubelet_size

        self.grid_height = img_size[0] // patch_size
        self.grid_width = img_size[1] // patch_size

        # ------------------------------------------------------------------ #
        # Tier 1 — Camera conditioning token encoders
        # All projections go to predictor_embed_dim (not embed_dim).
        # intrinsics_dim is free — no DROID `action_embed_dim - 1` hack.
        # ------------------------------------------------------------------ #
        self.state_encoder = nn.Linear(state_dim, predictor_embed_dim, bias=True)
        self.action_encoder = nn.Linear(action_dim, predictor_embed_dim, bias=True)
        if self.use_intrinsics:
            self.intrinsics_encoder = nn.Linear(intrinsics_dim, predictor_embed_dim, bias=True)

        # Number of conditioning tokens interleaved per frame:
        #   [action, state]         → cond_tokens = 2
        #   [action, state, K]      → cond_tokens = 3
        self.cond_tokens = 3 if self.use_intrinsics else 2

        # ------------------------------------------------------------------ #
        # Tier 2a — Multi-layer input embedding (V-JEPA 2.1 hierarchical)
        # If n_hierarchical_layers == 1, a plain linear projection.
        # If n_hierarchical_layers > 1, a small MLP compressor that accepts
        # the concatenation of L encoder-layer outputs.
        # ------------------------------------------------------------------ #
        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        if n_hierarchical_layers == 1:
            self.predictor_embed = nn.Linear(embed_dim, predictor_embed_dim, bias=True)
        else:
            self.predictor_embed = nn.Sequential(
                nn.Linear(embed_dim * n_hierarchical_layers, embed_dim, bias=True),
                act_layer_mlp(),
                nn.Linear(embed_dim, predictor_embed_dim, bias=True),
            )

        # ------------------------------------------------------------------ #
        # Transformer blocks — ACBlock for action-token-aware RoPE attention
        # ------------------------------------------------------------------ #
        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]
        self.predictor_blocks = nn.ModuleList(
            [
                Block(
                    use_rope=use_rope,
                    grid_size=self.grid_height,
                    dim=predictor_embed_dim,
                    num_heads=num_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    qk_scale=qk_scale,
                    drop=drop_rate,
                    act_layer=nn.SiLU if use_silu else nn.GELU,
                    wide_silu=wide_silu,
                    attn_drop=attn_drop_rate,
                    drop_path=dpr[i],
                    norm_layer=norm_layer,
                )
                for i in range(depth)
            ]
        )

        # ------------------------------------------------------------------ #
        # Tier 2b — Output projection heads (V-JEPA 2.1 hierarchical)
        # predictor_proj outputs n_hierarchical_layers * out_embed_dim so the
        # loss function can supervise each encoder layer independently.
        # ------------------------------------------------------------------ #
        if out_embed_dim is None:
            out_embed_dim = embed_dim
        self.out_embed_dim = out_embed_dim
        total_out_dim = n_hierarchical_layers * out_embed_dim

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.Linear(predictor_embed_dim, total_out_dim, bias=True)

        if self.predict_all:
            self.predictor_proj_context = nn.Linear(predictor_embed_dim, total_out_dim, bias=True)

        # ------------------------------------------------------------------ #
        # Weight initialisation + block rescaling (mirrors ac_predictor.py)
        # ------------------------------------------------------------------ #
        self.apply(self._init_weights)
        self._rescale_blocks()

        # Pre-compute and register the block-causal attention mask.
        # It is sliced to [:seq_len, :seq_len] at forward time so that clips
        # shorter than num_frames still work correctly.
        attn_mask = None
        if self.is_frame_causal:
            grid_depth = num_frames // tubelet_size
            attn_mask = build_action_block_causal_attention_mask(
                grid_depth,
                self.grid_height,
                self.grid_width,
                add_tokens=self.cond_tokens,
            )
        self.register_buffer("attn_mask", attn_mask, persistent=False)

    # ---------------------------------------------------------------------- #
    # Initialisation helpers
    # ---------------------------------------------------------------------- #

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=self.init_std)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def _rescale_blocks(self):
        def rescale(param, layer_id):
            param.div_(math.sqrt(2.0 * layer_id))

        for layer_id, layer in enumerate(self.predictor_blocks):
            rescale(layer.attn.proj.weight.data, layer_id + 1)
            rescale(layer.mlp.fc2.weight.data, layer_id + 1)

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x, actions, states, intrinsics=None):
        """
        Args:
            x: Visual context tokens from the (target) encoder.
               Shape: [B, T * H * W, embed_dim] for single-layer, or
                      [B, T * H * W, L * embed_dim] for L concatenated layers.
            actions: Relative camera motion (SE3 delta) per frame transition.
               Shape: [B, T, action_dim].
            states: Absolute camera pose per frame.
               Shape: [B, T, state_dim].
            intrinsics: Camera intrinsics per frame (optional).
               Shape: [B, T, intrinsics_dim].

        Returns:
            predictions: Projected frame-level predictions.
               Shape: [B, T * H * W, n_hierarchical_layers * out_embed_dim].
            context_predictions: Projected context-token predictions (only if
               predict_all=True, else None).
               Shape: [B, T * H * W, n_hierarchical_layers * out_embed_dim].
        """
        # -- project visual tokens to predictor dimension
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)

        # -- encode camera conditioning tokens → [B, T, 1, D] each
        a = self.action_encoder(actions).unsqueeze(2)    # [B, T, 1, D]
        s = self.state_encoder(states).unsqueeze(2)      # [B, T, 1, D]

        # -- reshape visual tokens for per-frame interleaving
        x_frames = x.view(B, T, self.grid_height * self.grid_width, D)  # [B, T, H*W, D]

        if self.use_intrinsics and intrinsics is not None:
            k = self.intrinsics_encoder(intrinsics).unsqueeze(2)  # [B, T, 1, D]
            # interleave: [action | state | intrinsics | patches] per frame
            x_seq = torch.cat([a, s, k, x_frames], dim=2).flatten(1, 2)  # [B, T*(3+H*W), D]
        else:
            # interleave: [action | state | patches] per frame
            x_seq = torch.cat([a, s, x_frames], dim=2).flatten(1, 2)     # [B, T*(2+H*W), D]

        # -- slice causal mask to actual sequence length
        seq_len = x_seq.size(1)
        attn_mask = self.attn_mask[:seq_len, :seq_len].to(x_seq.device, non_blocking=True)

        # -- transformer forward pass
        for blk in self.predictor_blocks:
            if self.use_activation_checkpointing:
                x_seq = torch.utils.checkpoint.checkpoint(
                    blk,
                    x_seq,
                    None,           # mask
                    attn_mask,
                    T,
                    self.grid_height,
                    self.grid_width,
                    self.cond_tokens,
                    use_reentrant=False,
                )
            else:
                x_seq = blk(
                    x_seq,
                    mask=None,
                    attn_mask=attn_mask,
                    T=T,
                    H=self.grid_height,
                    W=self.grid_width,
                    action_tokens=self.cond_tokens,
                )

        # -- split conditioning slots from visual patch slots
        #    layout per frame: [cond_0 .. cond_{K-1} | patch_0 .. patch_{HW-1}]
        x_seq = x_seq.view(B, T, self.cond_tokens + self.grid_height * self.grid_width, D)
        x_visual = x_seq[:, :, self.cond_tokens:, :].flatten(1, 2)   # [B, T*H*W, D]
        x_visual = self.predictor_norm(x_visual)

        predictions = self.predictor_proj(x_visual)  # [B, T*H*W, L*out_embed_dim]

        if self.predict_all:
            context_predictions = self.predictor_proj_context(x_visual)
            return predictions, context_predictions

        return predictions, None


def vit_camera_ac_predictor(**kwargs):
    model = CameraConditionedPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
