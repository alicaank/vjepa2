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
from typing import Optional, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F

from src.models.utils.modules import ACBlock as Block
from src.models.utils.modules import ACRoPEAttention
from src.models.utils.modules import build_action_block_causal_attention_mask
from src.utils.tensors import trunc_normal_

# FSQ categorical head lives in the GWM repo (not vjepa2). Imported lazily in
# __init__ so this module still imports cleanly when GWM's src/ is not on the
# path (e.g. when vjepa2 is used standalone).
try:  # pragma: no cover - best-effort import
    from src.training.common.fsq import FSQCategoricalHead  # type: ignore
except Exception:  # ModuleNotFoundError when GWM root is not on sys.path.
    FSQCategoricalHead = None  # type: ignore


class FourLayerBirthHead(nn.Module):
    """Per-layer masked residual completion head for CameraAC-LAM-4L.

    Replaces the single flattened ``completion_refine`` MLP that operated over
    all 4 layers concatenated (``Linear(8D, hidden, 4D)``).  That design is
    both structurally wrong (treats layer identity as spatial position) and
    expensive for ViT-G where D can be 1536+.

    Architecture
    ------------
    Shared cross-layer conditioner:
        input: [z_pred pooled over HW  (4D) | action_embed (action_dim) | mask_stats (n_mask_feats)]
        output: cond  (hidden)  — broadcast to every spatial position

    Per-layer head (× n_layers):
        input: [z_pred_l (D) | z_mem_l (D) | cond (hidden) | mask_feats_per_token (n_mask_feats)]
        output: residual_l (D)

    Final output: concat(residual_l, ...) shape (B, HW, n_layers * D)

    Gate
    ----
    ``completion_gate`` (scalar, init 0.0 → tanh(0) = 0): full no-op at
    step 0.  The caller adds ``tanh(gate) * output`` to ``z_pred``.
    """

    def __init__(
        self,
        layer_dim: int,
        action_dim: int,
        n_layers: int = 4,
        n_mask_feats: int = 4,
        hidden: int = 512,
        n_conditioner_layers: int = 2,
        n_head_layers: int = 2,
    ):
        super().__init__()
        self.n_layers = n_layers
        self.layer_dim = layer_dim

        cond_in = n_layers * layer_dim + action_dim + n_mask_feats
        cond_layers: list[nn.Module] = [nn.Linear(cond_in, hidden, bias=True), nn.GELU()]
        for _ in range(n_conditioner_layers - 1):
            cond_layers.extend([nn.Linear(hidden, hidden, bias=True), nn.GELU()])
        self.cross_layer_cond = nn.Sequential(*cond_layers)

        head_in = 2 * layer_dim + hidden + n_mask_feats
        heads = []
        for _ in range(n_layers):
            h_layers: list[nn.Module] = [nn.Linear(head_in, hidden, bias=True), nn.GELU()]
            for _ in range(n_head_layers - 1):
                h_layers.extend([nn.Linear(hidden, hidden, bias=True), nn.GELU()])
            h_layers.append(nn.Linear(hidden, layer_dim, bias=True))
            heads.append(nn.Sequential(*h_layers))
        self.heads = nn.ModuleList(heads)

        self.completion_gate = nn.Parameter(torch.zeros(()))
        # Zero-init the final residual projection in each per-layer head so
        # the birth head starts as a strict no-op: z_final = z_base + 0.
        # This prevents randomly-initialized residuals from destabilizing a
        # pre-trained model when completion is first turned on.
        for head in self.heads:
            last_linear = head[-1]
            nn.init.zeros_(last_linear.weight)
            nn.init.zeros_(last_linear.bias)

    def forward(
        self,
        z_pred: torch.Tensor,
        z_mem: torch.Tensor,
        mask_feats: torch.Tensor,
        action_embed: torch.Tensor,
    ) -> torch.Tensor:
        """
        Args:
            z_pred: (B, HW, n_layers * D)  raw predictor output
            z_mem:  (B, HW, n_layers * D)  warped/transported context features
                    (may be all-zeros when warp is off)
            mask_feats: (B, HW, n_mask_feats)  per-token region masks
                        [valid, boundary, reveal, transport_conf]
            action_embed: (B, action_dim)  incoming relative action features

        Returns:
            delta: (B, HW, n_layers * D) — caller applies
                   ``pred + tanh(gate) * delta``
        """
        B, HW, _ = z_pred.shape
        D = self.layer_dim

        pred = z_pred.view(B, HW, self.n_layers, D)
        mem = z_mem.view(B, HW, self.n_layers, D)

        pooled = pred.mean(dim=1).reshape(B, self.n_layers * D)
        mask_stats = mask_feats.mean(dim=1)
        cond_in = torch.cat([pooled, action_embed, mask_stats], dim=-1)
        cond = self.cross_layer_cond(cond_in)
        cond = cond.unsqueeze(1).expand(B, HW, -1)

        outs = []
        for l_idx, head in enumerate(self.heads):
            x = torch.cat([pred[:, :, l_idx], mem[:, :, l_idx], cond, mask_feats], dim=-1)
            outs.append(head(x))

        return torch.cat(outs, dim=-1)


class CameraUCPETargetAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int, gamma_init: float = 0.0, norm_layer=nn.LayerNorm):
        super().__init__()
        if dim % num_heads != 0:
            raise ValueError(f"dim={dim} must be divisible by num_heads={num_heads}")
        self.dim = int(dim)
        self.num_heads = int(num_heads)
        self.head_dim = self.dim // self.num_heads
        self.norm_q = norm_layer(dim)
        self.norm_kv = norm_layer(dim)
        self.q = nn.Linear(dim, dim, bias=True)
        self.kv = nn.Linear(dim, dim * 2, bias=True)
        self.proj = nn.Linear(dim, dim, bias=True)
        self.gamma = nn.Parameter(torch.tensor(float(gamma_init), dtype=torch.float32))

    def forward(self, x_seq, geom_seq, T: int, hw: int, cond_tokens: int):
        B, _, D = x_seq.shape
        target_start = (int(T) - 1) * (int(cond_tokens) + int(hw)) + int(cond_tokens)
        target_end = target_start + int(hw)
        target = x_seq[:, target_start:target_end, :]
        q = self.q(self.norm_q(target)).reshape(B, int(hw), self.num_heads, self.head_dim).transpose(1, 2)
        kv = self.kv(self.norm_kv(geom_seq)).reshape(B, geom_seq.size(1), 2, self.num_heads, self.head_dim)
        kv = kv.permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0, is_causal=False)
        out = out.transpose(1, 2).reshape(B, int(hw), D)
        out = self.proj(out)
        gamma = self.gamma.to(device=out.device, dtype=out.dtype)
        return torch.cat(
            [
                x_seq[:, :target_start, :],
                x_seq[:, target_start:target_end, :] + gamma * out,
                x_seq[:, target_end:, :],
            ],
            dim=1,
        )


class PredictorAdaLNModulation(nn.Module):
    """Per-block AdaLN modulation for the CameraAC predictor.

    Unlike DiT's AdaLN-Zero recipe, this path is not zero-initialized.  The
    residual gates are used as near-identity multipliers (1 + delta) so the
    block starts close to the normal ACBlock while still giving the camera
    action/pose condition a direct path into every transformer layer.
    """

    def __init__(self, dim: int, hidden: int, act_layer=nn.SiLU):
        super().__init__()
        hidden = max(1, int(hidden))
        self.net = nn.Sequential(
            nn.Linear(dim, hidden, bias=True),
            act_layer(),
            nn.Linear(hidden, 6 * dim, bias=True),
        )

    def forward(self, cond: torch.Tensor) -> tuple[torch.Tensor, ...]:
        return self.net(cond).chunk(6, dim=-1)


def _adaln_modulate(x: torch.Tensor, shift: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return x * (1.0 + scale.unsqueeze(1)) + shift.unsqueeze(1)


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
        use_ray_pe=False,
        ray_pe_dim=6,
        ray_pe_hidden=256,
        ray_pe_mode="origin_dir",
        ray_visibility_features="none",
        pose_conditioning_mode="token+raymap",
        action_token_mode="transition",
        use_delta_head=False,
        use_residual_head=False,
        residual_head_depth=2,
        residual_head_ratio=2.0,
        use_completion_head=False,
        completion_head_depth=2,
        completion_head_ratio=2.0,
        use_four_layer_birth_head=True,
        completion_head_hidden=512,
        completion_head_action_dim=None,
        use_fsq_head=False,
        fsq_total_code_axes=0,
        fsq_levels=0,
        warp_context_latents=False,
        warp_context_padding_mode="zeros",
        warp_mode="auto",
        depth_probe_checkpoint=None,
        target_slot_mode="mask",
        canvas_beta_init_valid=0.0,
        canvas_beta_init_boundary=0.0,
        canvas_beta_init_oov=0.0,
        canvas_use_mask_feat_embed=True,
        canvas_use_token_type_embed=True,
        canvas_use_target_step_embed=True,
        canvas_warp_mode="raw",
        canvas_projector_gamma_init=0.0,
        canvas_norm_clip_ratio=1.0,
        canvas_block_mask_enabled=False,
        canvas_block_mask_min_rects=1,
        canvas_block_mask_max_rects=3,
        canvas_block_mask_min_frac=0.2,
        canvas_block_mask_max_frac=0.3,
        correspondence_bias_enabled=False,
        correspondence_bias_mode="rotation_homography",
        correspondence_bias_sigma_tokens=2.0,
        correspondence_bias_lambda_init=0.0,
        correspondence_bias_learnable=True,
        correspondence_bias_apply_layers=(0, 1, 2, 3, 4, 5),
        camera_ucpe_enabled=False,
        camera_ucpe_apply_layers=(0, 3, 6, 9),
        camera_ucpe_gamma_init=0.0,
        coord_qk_enabled=False,
        coord_qk_apply_layers=(0, 3, 6, 9),
        coord_qk_hidden=256,
        coord_qk_freqs=6,
        coord_qk_gamma_init=0.1,
        ray_qk_enabled=False,
        ray_qk_apply_layers=(0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11),
        ray_qk_gamma_init=0.5,
        target_motion_query_enabled=False,
        target_motion_query_hidden=128,
        target_motion_query_gamma_init=0.1,
        predictor_adaln_enabled=False,
        predictor_adaln_hidden=512,
        **kwargs,
    ):
        super().__init__()

        self.is_frame_causal = is_frame_causal
        self.use_intrinsics = use_intrinsics and (intrinsics_dim > 0)
        self.use_ray_pe = use_ray_pe
        self.use_delta_head = use_delta_head
        self.use_residual_head = use_residual_head
        self.use_completion_head = bool(use_completion_head)
        self.use_fsq_head = bool(use_fsq_head)
        action_token_mode = str(action_token_mode).lower().strip()
        if action_token_mode not in ("transition", "target_relative"):
            raise ValueError(
                f"action_token_mode={action_token_mode!r} unsupported. "
                "Expected one of: 'transition', 'target_relative'."
            )
        if action_token_mode == "target_relative" and int(state_dim) < 7:
            raise ValueError(
                "action_token_mode='target_relative' requires 7D SE(3) states "
                "[tx, ty, tz, qx, qy, qz, qw]."
            )
        self.action_token_mode = action_token_mode
        # Rotation-only (infinity-plane) homography warp of context-frame
        # tokens into the target frame's 24x24 grid before interleaving
        # conditioning tokens. See ``_warp_context_tokens``.
        self.warp_context_latents = bool(warp_context_latents)
        _valid_padding_modes = ("border", "zeros", "reflection", "learned")
        if warp_context_padding_mode not in _valid_padding_modes:
            raise ValueError(
                f"warp_context_padding_mode must be one of {_valid_padding_modes}; "
                f"got {warp_context_padding_mode!r}."
            )
        self.warp_context_padding_mode = str(warp_context_padding_mode)
        # Path B-lite Step 1 — resolve ``warp_mode`` (one of
        #   ``"off"``               -- no token warping,
        #   ``"rotation_only"``     -- legacy rotation-only homography,
        #   ``"projective_probe"``  -- depth-aware projective warp using a
        #                              frozen DA3-distilled depth probe,
        #   ``"projective_da3"``    -- depth-aware projective warp using
        #                              externally supplied DA3 target depth).
        # The default ``"auto"`` derives the mode from ``warp_context_latents``
        # so every pre-Step-1 config / checkpoint produces bit-identical
        # behaviour: ``warp_context_latents=True`` ↔ ``"rotation_only"``;
        # ``warp_context_latents=False`` ↔ ``"off"``.
        if warp_mode == "auto":
            warp_mode = "rotation_only" if self.warp_context_latents else "off"
        if warp_mode not in ("off", "rotation_only", "projective_probe", "projective_da3"):
            raise ValueError(
                f"warp_mode must be one of {{off, rotation_only, projective_probe, projective_da3, auto}}; "
                f"got {warp_mode!r}."
            )
        self.warp_mode = str(warp_mode)
        if self.warp_mode != "off":
            # Force-enable the legacy flag so any downstream code that still
            # branches on it (eval scripts, attn-bias setup, etc.) treats the
            # projective_probe path as a warping run.
            self.warp_context_latents = True
        if self.warp_context_latents and not self.use_intrinsics:
            raise ValueError(
                "warp_context_latents=True / warp_mode!=off requires use_intrinsics=True "
                "so per-frame K is available to construct the homography."
            )

        # Frozen depth probe for ``warp_mode="projective_probe"``. Loaded once
        # at construction; the baked α scalar lives on ``_depth_probe_alpha``
        # (registered as a buffer so it round-trips through state_dict +
        # ``.to(device)``).
        #
        # Two probe architectures are supported (dispatched on the ``arch``
        # field in the checkpoint, defaulting to "simple" for backward
        # compatibility with Step 1-lite checkpoints that have no field):
        #
        #   * "simple"     — ``nn.Linear(feat_dim, 1)``  (Step 1-lite)
        #   * "multiscale" — ``MultiScaleResidualDepthProbe`` (Step 1-full v1)
        #
        # Both produce ``(B, HW)`` log-depth from ``(B, HW, feat_dim)`` raw
        # 4-layer concat features. The predictor exponentiates and scales
        # by ``_depth_probe_alpha`` in ``forward()``.
        self._depth_probe = None
        self._depth_probe_arch = None
        self._depth_probe_grid_hw: Optional[Tuple[int, int]] = None
        self._depth_probe_predicts_uncert = False
        self.register_buffer("_depth_probe_alpha", torch.tensor(1.0), persistent=True)
        if self.warp_mode == "projective_probe":
            if depth_probe_checkpoint is None:
                raise ValueError(
                    "warp_mode='projective_probe' requires depth_probe_checkpoint."
                )
            ckpt = torch.load(depth_probe_checkpoint, map_location="cpu", weights_only=False)
            ckpt_feat_dim = int(ckpt["feat_dim"])
            expected_feat_dim = int(embed_dim) * int(n_hierarchical_layers)
            if ckpt_feat_dim != expected_feat_dim:
                raise ValueError(
                    f"depth probe expects feat_dim={ckpt_feat_dim} but predictor "
                    f"input_token_dim={expected_feat_dim} "
                    f"(embed_dim={embed_dim} × n_hierarchical_layers={n_hierarchical_layers}). "
                    f"Re-train the probe to match, or check the encoder config."
                )
            arch = str(ckpt.get("arch", "simple")).lower().strip()
            self._depth_probe_arch = arch
            if arch == "simple":
                # Backward-compat: legacy state-dict has bare keys
                # {"weight": (1, F), "bias": (1,)} — wrap in a Linear.
                probe = nn.Linear(ckpt_feat_dim, 1, bias=True)
                probe.load_state_dict(ckpt["probe_state_dict"])
            elif arch == "multiscale":
                # Lazy import to avoid pulling tools/* into the predictor's
                # mandatory import path; only loaded when needed.
                from tools.depth_probe_models import (
                    MultiScaleResidualDepthProbe,
                )
                hidden_dim = int(ckpt.get("hidden_dim", 256))
                n_layers = int(ckpt.get("n_layers", n_hierarchical_layers))
                self._depth_probe_predicts_uncert = bool(
                    ckpt.get("predict_uncert", False)
                )
                probe = MultiScaleResidualDepthProbe(
                    feat_dim=ckpt_feat_dim,
                    n_layers=n_layers,
                    hidden_dim=hidden_dim,
                    predict_uncert=self._depth_probe_predicts_uncert,
                )
                probe.load_state_dict(ckpt["probe_state_dict"])
            else:
                raise ValueError(
                    f"depth probe checkpoint has unknown arch={arch!r}; "
                    f"expected 'simple' or 'multiscale'."
                )
            for p in probe.parameters():
                p.requires_grad_(False)
            probe.eval()
            self._depth_probe = probe
            self._depth_probe_alpha.data.fill_(float(ckpt["alpha_global"]))
            # Grid metadata for the multiscale forward (``simple`` ignores it
            # but we cache the ckpt's grid for assertions in forward).
            ckpt_gh = int(ckpt.get("grid_h", 0)) or None
            ckpt_gw = int(ckpt.get("grid_w", 0)) or None
            if ckpt_gh and ckpt_gw:
                self._depth_probe_grid_hw = (ckpt_gh, ckpt_gw)
        # Phase E1 — rotation-homography correspondence attention bias.
        # See ``@/home/ak/GWM/docs/CORRESPONDENCE_BIAS_E1.md`` for design.
        # Default off so every pre-correspondence run is bit-identical.
        self.correspondence_bias_enabled = bool(correspondence_bias_enabled)
        self.correspondence_bias_mode = str(correspondence_bias_mode)
        self.correspondence_bias_sigma_tokens = float(correspondence_bias_sigma_tokens)
        if self.correspondence_bias_enabled:
            if self.correspondence_bias_mode != "rotation_homography":
                raise NotImplementedError(
                    "Phase E1 supports correspondence_bias_mode='rotation_homography' "
                    f"only; got {self.correspondence_bias_mode!r}. Epipolar-line bias "
                    "(E2) is gated on E1 outcomes."
                )
            if not self.use_intrinsics:
                raise ValueError(
                    "correspondence_bias_enabled=True requires use_intrinsics=True "
                    "so per-frame K is available to construct the homography."
                )
            if self.correspondence_bias_sigma_tokens <= 0.0:
                raise ValueError(
                    f"correspondence_bias_sigma_tokens must be > 0; got "
                    f"{self.correspondence_bias_sigma_tokens}"
                )
        # ``apply_layers`` is validated against ``depth`` after ``predictor_blocks``
        # is built (see below). Store the raw tuple here.
        self._correspondence_bias_apply_layers_raw = tuple(int(i) for i in correspondence_bias_apply_layers)
        self._correspondence_bias_lambda_init = float(correspondence_bias_lambda_init)
        self._correspondence_bias_learnable = bool(correspondence_bias_learnable)
        self.camera_ucpe_enabled = bool(camera_ucpe_enabled)
        self._camera_ucpe_apply_layers_raw = tuple(int(i) for i in camera_ucpe_apply_layers)
        self.camera_ucpe_gamma_init = float(camera_ucpe_gamma_init)
        if self.camera_ucpe_enabled and not self.use_ray_pe:
            raise ValueError("camera_ucpe_enabled=True requires use_ray_pe=True so dense camera geometry tokens are available.")
        self.coord_qk_enabled = bool(coord_qk_enabled)
        self._coord_qk_apply_layers_raw = tuple(int(i) for i in coord_qk_apply_layers)
        self.coord_qk_hidden = int(coord_qk_hidden)
        self.coord_qk_freqs = int(coord_qk_freqs)
        self.coord_qk_gamma_init = float(coord_qk_gamma_init)
        if self.coord_qk_enabled:
            if not self.use_intrinsics:
                raise ValueError("coord_qk_enabled=True requires use_intrinsics=True.")
            if self.coord_qk_hidden <= 0:
                raise ValueError(f"coord_qk_hidden must be > 0; got {self.coord_qk_hidden}")
            if self.coord_qk_freqs <= 0:
                raise ValueError(f"coord_qk_freqs must be > 0; got {self.coord_qk_freqs}")
        self.ray_qk_enabled = bool(ray_qk_enabled)
        self._ray_qk_apply_layers_raw = tuple(int(i) for i in ray_qk_apply_layers)
        self.ray_qk_gamma_init = float(ray_qk_gamma_init)
        if self.ray_qk_enabled and not self.use_ray_pe:
            raise ValueError("ray_qk_enabled=True requires use_ray_pe=True.")
        self.target_motion_query_enabled = bool(target_motion_query_enabled)
        self.target_motion_query_hidden = int(target_motion_query_hidden)
        self.target_motion_query_gamma_init = float(target_motion_query_gamma_init)
        if self.target_motion_query_enabled:
            if not self.use_intrinsics:
                raise ValueError("target_motion_query_enabled=True requires use_intrinsics=True.")
            if self.target_motion_query_hidden <= 0:
                raise ValueError(
                    f"target_motion_query_hidden must be > 0; got {self.target_motion_query_hidden}"
                )
        self.predictor_adaln_enabled = bool(predictor_adaln_enabled)
        self.predictor_adaln_hidden = int(predictor_adaln_hidden)
        if self.predictor_adaln_enabled and not bool(use_rope):
            raise ValueError(
                "predictor_adaln_enabled=True is the AdaLN+RoPE v1 variant "
                "and requires use_rope=True."
            )
        self.fsq_total_code_axes = int(fsq_total_code_axes)
        self.fsq_levels = int(fsq_levels)
        self.predict_all = predict_all
        self.n_hierarchical_layers = n_hierarchical_layers
        self.input_token_dim = embed_dim * n_hierarchical_layers
        target_slot_mode = str(target_slot_mode).lower().strip()
        if target_slot_mode not in ("mask", "canvas_first"):
            raise ValueError(
                f"target_slot_mode must be one of ('mask', 'canvas_first'); got {target_slot_mode!r}."
            )
        self.target_slot_mode = target_slot_mode
        self.canvas_use_mask_feat_embed = bool(canvas_use_mask_feat_embed)
        self.canvas_use_token_type_embed = bool(canvas_use_token_type_embed)
        self.canvas_use_target_step_embed = bool(canvas_use_target_step_embed)
        self.canvas_warp_mode = str(canvas_warp_mode).lower().strip()
        if self.canvas_warp_mode not in ("raw", "projector", "norm_clip", "rot_gated"):
            raise ValueError(
                f"canvas_warp_mode must be one of ('raw', 'projector', 'norm_clip', 'rot_gated'); got {self.canvas_warp_mode!r}."
            )
        self.canvas_projector_gamma_init = float(canvas_projector_gamma_init)
        self.canvas_norm_clip_ratio = float(canvas_norm_clip_ratio)
        self.canvas_block_mask_enabled = bool(canvas_block_mask_enabled)
        self.canvas_block_mask_min_rects = max(1, int(canvas_block_mask_min_rects))
        self.canvas_block_mask_max_rects = max(self.canvas_block_mask_min_rects, int(canvas_block_mask_max_rects))
        self.canvas_block_mask_min_frac = max(0.0, min(1.0, float(canvas_block_mask_min_frac)))
        self.canvas_block_mask_max_frac = max(self.canvas_block_mask_min_frac, min(1.0, float(canvas_block_mask_max_frac)))
        self._canvas_beta_init = (
            float(canvas_beta_init_valid),
            float(canvas_beta_init_boundary),
            float(canvas_beta_init_oov),
        )
        self.use_activation_checkpointing = use_activation_checkpointing
        self.init_std = init_std

        if self.use_fsq_head:
            if self.fsq_total_code_axes <= 0 or self.fsq_levels <= 0:
                raise ValueError(
                    "use_fsq_head=True requires fsq_total_code_axes > 0 and "
                    f"fsq_levels > 0; got total_code_axes={self.fsq_total_code_axes} "
                    f"levels={self.fsq_levels}."
                )
            if FSQCategoricalHead is None:
                raise ImportError(
                    "CameraConditionedPredictorAC was built with use_fsq_head=True "
                    "but `src.training.common.fsq.FSQCategoricalHead` is not "
                    "importable. Ensure the GWM repo root is on sys.path."
                )

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

        # Pose-conditioning mode (DA3 RayMap v3, see
        # ``@/home/ak/GWM/docs/RAYMAP_V3_DESIGN.md``):
        #
        # * ``token+raymap`` (default, legacy): per-frame pose lives in a
        #   single dense state token ``s = state_encoder(states)`` plus the
        #   optional per-patch ray PE. This is what every prior pilot used.
        # * ``raymap_only``: drop the state token entirely. Per-frame pose
        #   information lives ONLY in the dense per-patch raymap PE added to
        #   the visual tokens. Forces ``use_ray_pe=True``. The ``state_encoder``
        #   layer stays defined so ``token+raymap`` checkpoints can still be
        #   loaded for cross-mode evaluation; the state token is simply not
        #   concatenated into ``x_seq``.
        #
        # Number of conditioning tokens interleaved per frame:
        #   token+raymap, use_intrinsics=True : [action, state, K]      → 3
        #   token+raymap, use_intrinsics=False: [action, state]          → 2
        #   raymap_only,  use_intrinsics=True : [action, K]              → 2
        #   raymap_only,  use_intrinsics=False: [action]                 → 1
        pose_conditioning_mode = str(pose_conditioning_mode).lower().strip()
        if pose_conditioning_mode not in ("token+raymap", "raymap_only", "raymap_dual"):
            raise ValueError(
                f"pose_conditioning_mode={pose_conditioning_mode!r} unsupported. "
                f"Expected one of: 'token+raymap', 'raymap_only', 'raymap_dual'."
            )
        if pose_conditioning_mode in ("raymap_only", "raymap_dual") and not bool(use_ray_pe):
            raise ValueError(
                f"pose_conditioning_mode={pose_conditioning_mode!r} requires use_ray_pe=True. "
                "Otherwise per-frame pose information is lost entirely (the "
                "state token is dropped and there is no ray PE to replace it)."
            )
        self.pose_conditioning_mode = pose_conditioning_mode
        # Number of conditioning tokens interleaved per frame:
        #   token+raymap, K=True : [action, state, K]    → 3
        #   token+raymap, K=False: [action, state]        → 2
        #   raymap_only,  K=True : [action, K]            → 2
        #   raymap_only,  K=False: [action]               → 1
        #   raymap_dual,  K=True : [K]                    → 1   (action also dropped)
        #   raymap_dual,  K=False: []                     → 0
        if pose_conditioning_mode == "raymap_only":
            self.cond_tokens = 2 if self.use_intrinsics else 1
        elif pose_conditioning_mode == "raymap_dual":
            self.cond_tokens = 1 if self.use_intrinsics else 0
        else:
            self.cond_tokens = 3 if self.use_intrinsics else 2

        # Ray-PE representation mode. ``origin_dir`` (default) preserves the
        # original implementation byte-for-byte: per-token feature is the raw
        # concatenation ``[ray_origin_world_anchor_relative, ray_dir_world]``
        # (6D). ``plucker`` uses the canonical Plücker line representation
        # ``[ray_dir_world, ray_origin × ray_dir_world]`` (6D), which is
        # invariant to the choice of point along the ray and rotation-
        # equivariant via ``R(o×d) = (Ro)×(Rd)``. ``plucker_delta`` extends
        # plucker with the per-frame delta vs the anchor (frame 0) ray at
        # the same grid location, doubling the input dim to 12; this gives
        # the predictor explicit "how did this token's ray move" signal.
        # See ``@/home/ak/GWM/docs/RAYMAP_V2_DESIGN.md`` for design rationale.
        #
        # ``plucker_pair`` (RayMap v4c): per-token feature pairs the current
        # frame's ray with the *previous* frame's ray at the same grid cell,
        # so the predictor sees explicit per-token camera displacement. The
        # concatenation is [d_t, m_t, d_{t-1}, m_{t-1}, Δo, Δd, d_t·d_{t-1},
        # d_t×d_{t-1}] = 22 dims. For frame 0 the "previous" frame is itself,
        # which yields zero deltas / unit dot / zero cross (well-defined,
        # geometrically degenerate but not broken). Optional visibility
        # features (controlled by ``ray_visibility_features``) add 4 dims
        # from a rotation-only warp of the target patch into the previous
        # frame: [uv_warp_x, uv_warp_y, valid_flag, border_distance]. See
        # ``@/home/ak/GWM/docs/CAMERA_PREDICTOR_FULL_HISTORY.md#4.6``.
        ray_pe_mode = str(ray_pe_mode).lower().strip()
        if ray_pe_mode not in ("origin_dir", "plucker", "plucker_delta", "plucker_pair"):
            raise ValueError(
                f"ray_pe_mode={ray_pe_mode!r} unsupported. "
                f"Expected one of: 'origin_dir', 'plucker', 'plucker_delta', 'plucker_pair'."
            )
        self.ray_pe_mode = ray_pe_mode

        # Visibility features (only meaningful for ``plucker_pair``). Adds
        # a 4-channel rotation-only-warp correspondence block to the per-
        # token feature: [uv_src_warped_x, uv_src_warped_y, valid_flag,
        # border_distance]. "warp_plus_border" is the standard setting.
        ray_visibility_features = str(ray_visibility_features).lower().strip()
        if ray_visibility_features not in ("none", "warp_plus_border"):
            raise ValueError(
                f"ray_visibility_features={ray_visibility_features!r} unsupported. "
                f"Expected one of: 'none', 'warp_plus_border'."
            )
        if ray_visibility_features != "none" and ray_pe_mode != "plucker_pair":
            raise ValueError(
                f"ray_visibility_features={ray_visibility_features!r} only valid "
                f"with ray_pe_mode='plucker_pair' (got {ray_pe_mode!r})."
            )
        self.ray_visibility_features = ray_visibility_features
        if self.use_ray_pe:
            # Validate ray_pe_dim matches the chosen representation. Cross-
            # check here keeps misconfigurations (e.g. plucker_delta with
            # ray_pe_dim=6) from silently producing wrong shapes downstream.
            expected_dim = {
                "origin_dir": 6,
                "plucker": 6,
                "plucker_delta": 12,
                "plucker_pair": 22,
            }[ray_pe_mode]
            if self.ray_visibility_features == "warp_plus_border":
                expected_dim += 4
            if int(ray_pe_dim) != expected_dim:
                raise ValueError(
                    f"ray_pe_mode={ray_pe_mode!r} + ray_visibility_features="
                    f"{ray_visibility_features!r} requires ray_pe_dim={expected_dim} "
                    f"(got ray_pe_dim={ray_pe_dim})."
                )
            self.ray_pe_mlp = nn.Sequential(
                nn.Linear(ray_pe_dim, ray_pe_hidden, bias=True),
                nn.GELU(),
                nn.Linear(ray_pe_hidden, predictor_embed_dim, bias=True),
            )
            # Action raymap MLP (only used for pose_conditioning_mode='raymap_dual').
            # Encodes the per-patch effect of the relative-pose action: each
            # patch's camera-local direction is rotated by R_action and paired
            # with the action translation, producing a 6D Plücker line that
            # describes "how this token's ray transforms under the action".
            # Parameter shape mirrors ray_pe_mlp; weights are independent so
            # state and action signals do not share a feature MLP.
            if self.pose_conditioning_mode == "raymap_dual":
                self.action_ray_pe_mlp = nn.Sequential(
                    nn.Linear(6, ray_pe_hidden, bias=True),
                    nn.GELU(),
                    nn.Linear(ray_pe_hidden, predictor_embed_dim, bias=True),
                )
            gh, gw = self.grid_height, self.grid_width
            cy_grid = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
            cx_grid = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
            yy, xx = torch.meshgrid(cy_grid, cx_grid, indexing="ij")
            self.register_buffer(
                "_ray_pixel_grid",
                torch.stack([xx, yy], dim=-1).reshape(-1, 2),
                persistent=False,
            )
        elif self.coord_qk_enabled:
            gh, gw = self.grid_height, self.grid_width
            cy_grid = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
            cx_grid = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
            yy, xx = torch.meshgrid(cy_grid, cx_grid, indexing="ij")
            self.register_buffer(
                "_ray_pixel_grid",
                torch.stack([xx, yy], dim=-1).reshape(-1, 2),
                persistent=False,
            )
        elif self.target_motion_query_enabled:
            gh, gw = self.grid_height, self.grid_width
            cy_grid = (torch.arange(gh, dtype=torch.float32) + 0.5) / gh
            cx_grid = (torch.arange(gw, dtype=torch.float32) + 0.5) / gw
            yy, xx = torch.meshgrid(cy_grid, cx_grid, indexing="ij")
            self.register_buffer(
                "_ray_pixel_grid",
                torch.stack([xx, yy], dim=-1).reshape(-1, 2),
                persistent=False,
            )
        # ------------------------------------------------------------------ #
        # Tier 2a — Multi-layer input embedding (V-JEPA 2.1 hierarchical)
        # If n_hierarchical_layers == 1, a plain linear projection.
        # If n_hierarchical_layers > 1, a small MLP compressor that accepts
        # the concatenation of L encoder-layer outputs.
        # ------------------------------------------------------------------ #
        act_layer_mlp = nn.SiLU if use_silu else nn.GELU
        self.future_mask_token = nn.Parameter(torch.zeros(1, 1, self.input_token_dim))
        if self.target_slot_mode == "canvas_first":
            self.canvas_beta_by_type = nn.Parameter(torch.tensor(self._canvas_beta_init, dtype=torch.float32))
            self.canvas_mask_proj = nn.Linear(4, self.input_token_dim, bias=True)
            self.canvas_type_embed = nn.Embedding(3, self.input_token_dim)
            self.canvas_target_step_embed = nn.Embedding(max(1, num_frames // tubelet_size), self.input_token_dim)
            if self.canvas_warp_mode == "projector":
                self.canvas_warp_norm = nn.LayerNorm(self.input_token_dim)
                self.canvas_warp_proj = nn.Linear(self.input_token_dim, self.input_token_dim, bias=True)
                self.canvas_warp_gamma = nn.Parameter(torch.tensor(self.canvas_projector_gamma_init, dtype=torch.float32))
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
        if self.predictor_adaln_enabled:
            # Pool sparse camera tokens once per clip, project to the predictor
            # width, then give every transformer block its own AdaLN head.
            cond_inputs = 3 if self.use_intrinsics else 2
            self.predictor_adaln_conditioner = nn.Sequential(
                norm_layer(cond_inputs * predictor_embed_dim),
                nn.Linear(cond_inputs * predictor_embed_dim, predictor_embed_dim, bias=True),
                nn.SiLU(),
                nn.Linear(predictor_embed_dim, predictor_embed_dim, bias=True),
            )
            self.predictor_adaln_modulators = nn.ModuleList(
                [
                    PredictorAdaLNModulation(
                        dim=predictor_embed_dim,
                        hidden=self.predictor_adaln_hidden,
                        act_layer=nn.SiLU,
                    )
                    for _ in range(depth)
                ]
            )
        if self.camera_ucpe_enabled:
            apply_set = set(self._camera_ucpe_apply_layers_raw)
            invalid = [i for i in apply_set if i < 0 or i >= depth]
            if invalid:
                raise ValueError(
                    f"camera_ucpe_apply_layers entries must be in [0, {depth}); got out-of-range indices {sorted(invalid)}"
                )
            self.camera_ucpe_apply_layers = tuple(sorted(apply_set))
            self.camera_ucpe_branches = nn.ModuleDict(
                {
                    str(i): CameraUCPETargetAttention(
                        dim=predictor_embed_dim,
                        num_heads=num_heads,
                        gamma_init=self.camera_ucpe_gamma_init,
                        norm_layer=norm_layer,
                    )
                    for i in self.camera_ucpe_apply_layers
                }
            )
        else:
            self.camera_ucpe_apply_layers = tuple()

        if self.coord_qk_enabled:
            apply_set = set(self._coord_qk_apply_layers_raw)
            invalid = [i for i in apply_set if i < 0 or i >= depth]
            if invalid:
                raise ValueError(
                    f"coord_qk_apply_layers entries must be in [0, {depth}); got out-of-range indices {sorted(invalid)}"
                )
            self.coord_qk_apply_layers = tuple(sorted(apply_set))
            coord_in_dim = 3 + 2 * 3 * self.coord_qk_freqs
            self.coord_qk_mlp = nn.Sequential(
                nn.Linear(coord_in_dim, self.coord_qk_hidden, bias=True),
                nn.GELU(),
                nn.Linear(self.coord_qk_hidden, predictor_embed_dim, bias=True),
            )
            self.coord_qk_gamma = nn.Parameter(
                torch.full((len(self.coord_qk_apply_layers),), self.coord_qk_gamma_init, dtype=torch.float32)
            )
        else:
            self.coord_qk_apply_layers = tuple()
        if self.ray_qk_enabled:
            apply_set = set(self._ray_qk_apply_layers_raw)
            invalid = [i for i in apply_set if i < 0 or i >= depth]
            if invalid:
                raise ValueError(
                    f"ray_qk_apply_layers entries must be in [0, {depth}); got out-of-range indices {sorted(invalid)}"
                )
            self.ray_qk_apply_layers = tuple(sorted(apply_set))
            self.ray_qk_gamma = nn.Parameter(
                torch.full((len(self.ray_qk_apply_layers),), self.ray_qk_gamma_init, dtype=torch.float32)
            )
        else:
            self.ray_qk_apply_layers = tuple()
        if self.target_motion_query_enabled:
            self.target_motion_query_mlp = nn.Sequential(
                nn.Linear(11, self.target_motion_query_hidden, bias=True),
                nn.GELU(),
                nn.Linear(self.target_motion_query_hidden, predictor_embed_dim, bias=True),
            )
            self.target_motion_query_gamma = nn.Parameter(
                torch.tensor(self.target_motion_query_gamma_init, dtype=torch.float32)
            )

        # ------------------------------------------------------------------ #
        # Tier 2b — Output projection heads (V-JEPA 2.1 hierarchical)
        # predictor_proj outputs n_hierarchical_layers * out_embed_dim so the
        # loss function can supervise each encoder layer independently.
        # ------------------------------------------------------------------ #
        if out_embed_dim is None:
            out_embed_dim = embed_dim
        self.out_embed_dim = out_embed_dim

        self.predictor_norm = norm_layer(predictor_embed_dim)
        self.predictor_proj = nn.ModuleList(
            [nn.Linear(predictor_embed_dim, out_embed_dim, bias=True) for _ in range(n_hierarchical_layers)]
        )

        if self.predict_all:
            self.predictor_proj_context = nn.ModuleList(
                [nn.Linear(predictor_embed_dim, out_embed_dim, bias=True) for _ in range(n_hierarchical_layers)]
            )

        if self.use_delta_head:
            self.predictor_proj_delta = nn.ModuleList(
                [nn.Linear(predictor_embed_dim, out_embed_dim, bias=True) for _ in range(n_hierarchical_layers)]
            )

        # ------------------------------------------------------------------ #
        # Phase E1 — finalize correspondence-bias per-layer state. Must run
        # after ``predictor_blocks`` so we can validate ``apply_layers`` and
        # know ``depth``.
        # ------------------------------------------------------------------ #
        if self.correspondence_bias_enabled:
            apply_set = set(self._correspondence_bias_apply_layers_raw)
            invalid = [i for i in apply_set if i < 0 or i >= depth]
            if invalid:
                raise ValueError(
                    f"correspondence_bias_apply_layers entries must be in "
                    f"[0, {depth}); got out-of-range indices {sorted(invalid)}"
                )
            self.correspondence_bias_apply_layers = tuple(sorted(apply_set))
            n_apply = len(self.correspondence_bias_apply_layers)
            init = self._correspondence_bias_lambda_init
            lam = torch.full((n_apply,), init, dtype=torch.float32)
            if self._correspondence_bias_learnable:
                self.correspondence_bias_lambda = nn.Parameter(lam)
            else:
                self.register_buffer(
                    "correspondence_bias_lambda", lam, persistent=True,
                )
            # Per-epoch ramp scalar in [0, 1]. Trainer sets this each epoch
            # (mirroring Phase R cycle weight). Default 1.0 so eval / inference
            # paths that never call the trainer see the full prior.
            self.register_buffer(
                "correspondence_bias_ramp",
                torch.ones((), dtype=torch.float32),
                persistent=False,
            )
        else:
            self.correspondence_bias_apply_layers = tuple()

        if self.use_fsq_head:
            # Categorical head that maps post-residual hidden states to
            # (B, N, total_code_axes, levels) logits. The continuous heads
            # above stay intact; callers choose which to decode from.
            self.fsq_head = FSQCategoricalHead(
                predictor_embed_dim=predictor_embed_dim,
                total_code_axes=self.fsq_total_code_axes,
                levels=self.fsq_levels,
            )

        if self.use_residual_head:
            residual_hidden = int(predictor_embed_dim * residual_head_ratio)
            layers = []
            in_dim = predictor_embed_dim
            for _ in range(residual_head_depth):
                layers.extend(
                    [
                        nn.Linear(in_dim, residual_hidden, bias=True),
                        nn.GELU(),
                    ]
                )
                in_dim = residual_hidden
            layers.append(nn.Linear(in_dim, predictor_embed_dim, bias=True))
            self.residual_refine = nn.Sequential(*layers)
            self.residual_gate = nn.Parameter(torch.tensor(0.01))

        self.use_four_layer_birth_head = bool(use_four_layer_birth_head)
        if self.use_completion_head:
            completion_dim = int(n_hierarchical_layers) * int(out_embed_dim)
            if self.use_four_layer_birth_head:
                # P2: structured per-layer head.  action_dim falls back to the
                # constructor's action_dim when not separately specified.
                _birth_action_dim = int(
                    completion_head_action_dim
                    if completion_head_action_dim is not None
                    else action_dim
                )
                _birth_hidden = int(completion_head_hidden)
                self.completion_refine = FourLayerBirthHead(
                    layer_dim=int(out_embed_dim),
                    action_dim=_birth_action_dim,
                    n_layers=int(n_hierarchical_layers),
                    n_mask_feats=4,
                    hidden=_birth_hidden,
                    n_conditioner_layers=max(1, int(completion_head_depth) - 1),
                    n_head_layers=int(completion_head_depth),
                )
                # completion_gate lives inside FourLayerBirthHead; expose as
                # an attribute alias so legacy code that reads
                # ``predictor_ref.completion_gate`` still works.
                self.completion_gate = self.completion_refine.completion_gate
            else:
                # Legacy flat MLP path (backward compat, use_four_layer_birth_head=False).
                completion_hidden = int(completion_dim * completion_head_ratio)
                layers = []
                in_dim = completion_dim * 2
                for _ in range(int(completion_head_depth)):
                    layers.extend(
                        [
                            nn.Linear(in_dim, completion_hidden, bias=True),
                            nn.GELU(),
                        ]
                    )
                    in_dim = completion_hidden
                layers.append(nn.Linear(in_dim, completion_dim, bias=True))
                self.completion_refine = nn.Sequential(*layers)
                self.completion_gate = nn.Parameter(torch.tensor(0.01))

        # ------------------------------------------------------------------ #
        # Weight initialisation + block rescaling (mirrors ac_predictor.py)
        # ------------------------------------------------------------------ #
        self.apply(self._init_weights)
        trunc_normal_(self.future_mask_token, std=self.init_std)
        if self.target_slot_mode == "canvas_first":
            with torch.no_grad():
                self.canvas_beta_by_type.copy_(torch.tensor(self._canvas_beta_init, dtype=self.canvas_beta_by_type.dtype))
                self.canvas_mask_proj.weight.zero_()
                self.canvas_mask_proj.bias.zero_()
                self.canvas_type_embed.weight.zero_()
                self.canvas_target_step_embed.weight.zero_()
                if self.canvas_warp_mode == "projector":
                    self.canvas_warp_gamma.fill_(self.canvas_projector_gamma_init)
        if self.camera_ucpe_enabled:
            with torch.no_grad():
                for branch in self.camera_ucpe_branches.values():
                    branch.gamma.fill_(self.camera_ucpe_gamma_init)
        if self.coord_qk_enabled:
            with torch.no_grad():
                self.coord_qk_gamma.fill_(self.coord_qk_gamma_init)
        if self.ray_qk_enabled:
            with torch.no_grad():
                self.ray_qk_gamma.fill_(self.ray_qk_gamma_init)
        if self.target_motion_query_enabled:
            with torch.no_grad():
                self.target_motion_query_gamma.fill_(self.target_motion_query_gamma_init)
                self.target_motion_query_mlp[-1].weight.normal_(mean=0.0, std=1.0e-3)
                self.target_motion_query_mlp[-1].bias.zero_()
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

        # P0.4: learned unseen token for OOV warp regions.
        # When ``warp_context_padding_mode='learned'``, positions that project
        # outside the source frame grid are filled with this token instead of
        # being border-replicated or left as zeros.  Shape (1, 1, input_token_dim)
        # so it broadcasts to (B, HW_oov, D) after indexing.
        # Registered as nn.Parameter (not a buffer) so the optimizer updates it.
        if self.warp_context_padding_mode == "learned":
            self.warp_unseen_token = nn.Parameter(
                torch.zeros(1, 1, self.input_token_dim)
            )
            trunc_normal_(self.warp_unseen_token, std=0.02)
        else:
            self.warp_unseen_token = None
        # Side-channel: last OOV mask written by _apply_learned_oov.
        # Shape (B, gh, gw); None until first forward.
        # Consumed by train.py to build mask_feats for FourLayerBirthHead.
        self._last_warp_oov_mask: torch.Tensor | None = None
        self._last_canvas_diag: dict | None = None

    def _apply_learned_oov(
        self,
        warped_nchw: torch.Tensor,
        src_xy_gs: torch.Tensor,
    ) -> torch.Tensor:
        """Substitute ``warp_unseen_token`` into OOV (out-of-view) positions.

        Args:
            warped_nchw: ``(B, D, gh, gw)`` output of ``F.grid_sample`` with
                ``padding_mode='zeros'``.  OOV positions are all-zero.
            src_xy_gs: ``(B, gh, gw, 2)`` grid-sample coordinates in ``[-1,1]``.
                Positions with ``|x| > 1`` or ``|y| > 1`` are OOV.

        Returns:
            ``warped_nchw`` with OOV positions replaced by ``warp_unseen_token``.
            No-op when ``warp_unseen_token is None`` (padding_mode != 'learned').
        """
        B, D, gh, gw = warped_nchw.shape
        # Catch both out-of-range coords AND NaN projections (PyTorch documents
        # that NaN grid values are treated as -1, silently sampling the top-left
        # corner rather than being assigned the unseen token).
        finite = torch.isfinite(src_xy_gs).all(dim=-1)          # (B, gh, gw)
        oov = (~finite) | (src_xy_gs.abs() > 1.0).any(dim=-1)   # (B, gh, gw)
        # Expose for downstream mask_feats construction (detached — diagnostic
        # only, must not carry gradients into the loss through this path).
        self._last_warp_oov_mask = oov.detach()
        if self.warp_unseen_token is None:
            return warped_nchw
        if not oov.any():
            return warped_nchw
        unseen = self.warp_unseen_token.to(
            device=warped_nchw.device, dtype=warped_nchw.dtype
        )  # (1, 1, input_token_dim) — slice to D then broadcast
        unseen_d = unseen[..., :D].reshape(1, D, 1, 1).expand(B, D, gh, gw)
        oov_mask = oov.unsqueeze(1).expand(B, D, gh, gw)  # (B, D, gh, gw)
        return torch.where(oov_mask, unseen_d, warped_nchw)

    def complete_predictions(
        self,
        preds: torch.Tensor,
        warped_context_raw: torch.Tensor | None = None,
        mask_feats: torch.Tensor | None = None,
        action_embed: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Apply the completion-route residual for unseen/boundary tokens.

        Dispatches to ``FourLayerBirthHead`` when ``use_four_layer_birth_head=True``
        (default), or to the legacy flat-MLP path otherwise.

        The flat-MLP path ignores ``mask_feats`` and ``action_embed`` for
        backward compatibility with checkpoints that used the old interface.

        For the four-layer path, ``mask_feats`` should be ``(B, HW, 4)``
        containing [valid, boundary, reveal, transport_conf] per token.
        ``action_embed`` should be ``(B, action_dim)``.  Both default to zeros
        when not provided so the API remains callable from legacy code.
        """
        if not self.use_completion_head:
            return preds
        if warped_context_raw is None:
            warped_context_raw = torch.zeros_like(preds)
        else:
            warped_context_raw = warped_context_raw.to(device=preds.device, dtype=preds.dtype)
        if self.use_four_layer_birth_head:
            B, HW = preds.shape[:2]
            if mask_feats is None:
                mask_feats = preds.new_zeros(B, HW, 4)
            else:
                mask_feats = mask_feats.to(device=preds.device, dtype=preds.dtype)
            if action_embed is None:
                _act_dim = self.completion_refine.cross_layer_cond[0].in_features \
                    - self.completion_refine.n_layers * self.completion_refine.layer_dim \
                    - 4  # n_mask_feats
                action_embed = preds.new_zeros(B, _act_dim)
            else:
                action_embed = action_embed.to(device=preds.device, dtype=preds.dtype)
            delta = self.completion_refine(preds, warped_context_raw, mask_feats, action_embed)
            return preds + torch.tanh(self.completion_gate) * delta
        else:
            inp = torch.cat([preds, warped_context_raw], dim=-1)
            return preds + torch.tanh(self.completion_gate) * self.completion_refine(inp)

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

    @staticmethod
    def _build_K_from_normalized(K4: torch.Tensor) -> torch.Tensor:
        """Build (B, 3, 3) intrinsics matrices from normalized (fx, fy, cx, cy).

        Args:
            K4: [..., 4] with entries ``[fx/W, fy/H, cx/W, cy/H]`` (see the
                intrinsics layout used in ``_compute_ray_pe`` and the
                RE10KSequenceDataset loader).

        Returns:
            ``K`` that maps normalized image-coord rays ``(u/W, v/H, 1)`` to
            pixel locations on the unit-square [0, 1] image plane. I.e.
            ``K @ [x/z, y/z, 1]^T = [u_norm, v_norm, 1]^T``.
        """
        *lead, _ = K4.shape
        K = K4.new_zeros(*lead, 3, 3)
        K[..., 0, 0] = K4[..., 0]
        K[..., 1, 1] = K4[..., 1]
        K[..., 0, 2] = K4[..., 2]
        K[..., 1, 2] = K4[..., 3]
        K[..., 2, 2] = 1.0
        return K

    @staticmethod
    def _build_K_inv_from_normalized(K4: torch.Tensor) -> torch.Tensor:
        """Analytic inverse of the K matrix built by ``_build_K_from_normalized``.

        ``K^{-1} = [[1/fx, 0, -cx/fx], [0, 1/fy, -cy/fy], [0, 0, 1]]``.
        Uses ``clamp_min`` to avoid a /0 when focal lengths collapse to 0
        (shouldn't happen with real RE10K intrinsics but keeps gradients
        finite during the first few iterations before training finds its feet).
        """
        *lead, _ = K4.shape
        fx = K4[..., 0].clamp_min(1.0e-6)
        fy = K4[..., 1].clamp_min(1.0e-6)
        cx = K4[..., 2]
        cy = K4[..., 3]
        K_inv = K4.new_zeros(*lead, 3, 3)
        K_inv[..., 0, 0] = 1.0 / fx
        K_inv[..., 1, 1] = 1.0 / fy
        K_inv[..., 0, 2] = -cx / fx
        K_inv[..., 1, 2] = -cy / fy
        K_inv[..., 2, 2] = 1.0
        return K_inv

    def _prepare_target_depth(
        self,
        target_depth: torch.Tensor,
        B: int,
        HW: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor:
        depth_t = target_depth.to(device=device, dtype=dtype)
        if depth_t.dim() == 4 and depth_t.shape[1] == 1:
            depth_t = depth_t[:, 0]
        if depth_t.dim() == 3:
            if depth_t.shape[-2:] != (self.grid_height, self.grid_width):
                depth_t = F.interpolate(
                    depth_t.unsqueeze(1).float(),
                    size=(self.grid_height, self.grid_width),
                    mode="bilinear",
                    align_corners=False,
                ).to(device=device, dtype=dtype)[:, 0]
            depth_t = depth_t.reshape(B, HW)
        elif depth_t.dim() == 2:
            if depth_t.shape != (B, HW):
                raise ValueError(f"target_depth shape={tuple(depth_t.shape)} expected {(B, HW)}")
        else:
            raise ValueError(f"target_depth must be (B,HW), (B,H,W), or (B,1,H,W); got {tuple(depth_t.shape)}")
        return depth_t.clamp_min(1.0e-3)

    def _warp_context_tokens(
        self,
        x_embed: torch.Tensor,
        states: torch.Tensor,
        intrinsics: torch.Tensor,
        target_depth: torch.Tensor = None,
        depth_gate: torch.Tensor = None,
    ) -> torch.Tensor:
        """Rotation-only or depth-aware projective warp of context-frame
        tokens into the target frame's pixel grid.

        Dispatch is controlled by ``self.warp_mode``:

        * ``"rotation_only"`` (default for legacy ``warp_context_latents=True``)
          uses the infinity-plane homography ``H = K_c R_c^T R_t K_t^{-1}``.
          ``target_depth`` is ignored.
        * ``"projective_probe"`` uses per-target-pixel depth ``D_t`` to back-
          project to a 3D point, transforms it through both cameras, and
          reads the source pixel.  ``target_depth`` must be provided as
          ``(B, HW)`` in pose-consistent units (the depth probe's α-baked
          output already satisfies this; see ``__init__``).

        The α=∞ limit of the projective branch reproduces the rotation-only
        result, so the projective path is strictly more general.

        For each context frame ``c`` and target frame ``t = T - 1``:
            * Build relative rotation ``R_{c -> t} = R_t^T R_c`` from the
              xyzw quaternions in ``states`` (R_c maps cam-local -> anchor,
              so the relative rotation is invariant under the shared anchor).
            * Compose the infinity-plane homography
              ``H_{t -> c} = K_c R_c^T R_t K_t^{-1}`` which maps a target-
              frame pixel to the cam_c pixel to sample from.
            * Reshape the context token map to ``(B, D, gh, gw)`` and sample
              with ``F.grid_sample`` (bilinear, border padding by default).

        The target frame (last slot) is left untouched -- its tokens are
        either the future_mask_token (training) or don't carry spatial
        information that the rotation homography could meaningfully move.
        The approximation is exact for purely-rotational motion / infinitely
        distant scenes; for RE10K's room-scale translation it's a coarse
        prior at best but still pixel-aligns the dominant rotation component.

        Args:
            x_embed: ``(B, T * HW, D)`` post-``predictor_embed`` tokens.
            states: ``(B, T, 7)`` = ``[tx, ty, tz, qx, qy, qz, qw]``.
            intrinsics: ``(B, T, 4)`` = normalized ``[fx/W, fy/H, cx/W, cy/H]``.

        Returns:
            ``(B, T * HW, D)`` with context frames warped, target unchanged.
        """
        B, N_ctxt, D = x_embed.shape
        gh, gw = self.grid_height, self.grid_width
        HW = gh * gw
        T = N_ctxt // HW
        if T < 2:
            return x_embed

        # Target frame geometry (shared across all context frames).
        dtype = x_embed.dtype
        R_t = self._quat_xyzw_to_rotmat(
            states[:, -1, 3:7].to(dtype=dtype)
        )  # (B, 3, 3)
        K_t_inv = self._build_K_inv_from_normalized(
            intrinsics[:, -1].to(dtype=dtype)
        )  # (B, 3, 3)

        # Target pixel centers in normalized [0, 1] x [0, 1] coords,
        # matching the intrinsics convention (``_compute_ray_pe`` uses
        # ``(idx + 0.5) / gh`` for cell centers).
        y_norm = (torch.arange(gh, device=x_embed.device, dtype=dtype) + 0.5) / gh
        x_norm = (torch.arange(gw, device=x_embed.device, dtype=dtype) + 0.5) / gw
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        # Homogeneous target-frame pixel grid (HW, 3).
        pixel_grid_t = torch.stack(
            [xx.reshape(-1), yy.reshape(-1), torch.ones(HW, device=x_embed.device, dtype=dtype)],
            dim=-1,
        )  # (HW, 3)

        # View x as (B, T, HW, D) and process context frames only.
        x_frames = x_embed.view(B, T, HW, D)
        warped = [x_frames[:, c] for c in range(T)]  # start with identity

        # Pre-compute the back-projected target points in the anchor (world)
        # frame once per forward — only needed for the projective branch.
        # ``pts_world[b, n] = R_t @ (D_t[b, n] * K_t^{-1} u_t[n]) + t_t[b]``.
        pts_world = None
        if self.warp_mode in ("projective_probe", "projective_da3") and target_depth is not None:
            rays_cam_t = torch.einsum("bij,nj->bni", K_t_inv, pixel_grid_t)  # (B, HW, 3)
            depth_t = self._prepare_target_depth(target_depth, B, HW, x_embed.device, dtype)
            pts_cam_t = rays_cam_t * depth_t.unsqueeze(-1)                   # (B, HW, 3)
            t_t_vec = states[:, -1, :3].to(dtype=dtype).unsqueeze(1)         # (B, 1, 3)
            pts_world = torch.einsum("bij,bnj->bni", R_t, pts_cam_t) + t_t_vec  # (B, HW, 3)

        for c in range(T - 1):
            R_c = self._quat_xyzw_to_rotmat(
                states[:, c, 3:7].to(dtype=dtype)
            )  # (B, 3, 3)
            K_c = self._build_K_from_normalized(
                intrinsics[:, c].to(dtype=dtype)
            )  # (B, 3, 3)

            # Compute the rotation-only homography for this context frame
            # once; we may need it as the fallback path under uncertainty
            # gating even if projective is the primary mode.
            R_tc = torch.bmm(R_c.transpose(-1, -2), R_t)              # (B, 3, 3)
            H_inv = torch.bmm(K_c, torch.bmm(R_tc, K_t_inv))          # (B, 3, 3)
            src_h_rot = torch.einsum("bij,nj->bni", H_inv, pixel_grid_t)

            if self.warp_mode in ("projective_probe", "projective_da3") and pts_world is not None:
                t_c_vec = states[:, c, :3].to(dtype=dtype).unsqueeze(1)       # (B, 1, 3)
                pts_cam_c = torch.einsum(
                    "bji,bnj->bni", R_c, pts_world - t_c_vec
                )                                                              # (B, HW, 3)
                src_h_proj = torch.einsum("bij,bnj->bni", K_c, pts_cam_c)
                src_h = src_h_proj
            else:
                src_h = src_h_rot
            src_xy = src_h[..., :2] / src_h[..., 2:3].clamp_min(1.0e-6)  # (B, HW, 2)

            # grid_sample expects coords in [-1, 1] with (x, y) order.
            src_xy_gs = (src_xy * 2.0 - 1.0).reshape(B, gh, gw, 2)
            ctx_nchw = x_frames[:, c].reshape(B, gh, gw, D).permute(0, 3, 1, 2).contiguous()
            # When padding_mode='learned', pass 'zeros' to F.grid_sample so OOV
            # positions are zero-filled, then substitute the learned unseen token.
            _pm = "zeros" if self.warp_context_padding_mode == "learned" else self.warp_context_padding_mode
            warped_nchw = F.grid_sample(
                ctx_nchw,
                src_xy_gs,
                mode="bilinear",
                padding_mode=_pm,
                align_corners=False,
            )
            warped_nchw = self._apply_learned_oov(warped_nchw, src_xy_gs)
            warped_proj = warped_nchw.permute(0, 2, 3, 1).reshape(B, HW, D)

            if (
                depth_gate is not None
                and self.warp_mode in ("projective_probe", "projective_da3")
            ):
                # Per-token blend: high-confidence (large gate) tokens use
                # projective transport; low-confidence tokens fall back to
                # the rotation-only homography. This is the v1.5 mechanism
                # — see history doc §9.6 (uncertainty + warp gating).
                src_xy_rot = src_h_rot[..., :2] / src_h_rot[..., 2:3].clamp_min(1.0e-6)
                src_xy_rot_gs = (src_xy_rot * 2.0 - 1.0).reshape(B, gh, gw, 2)
                warped_rot_nchw = F.grid_sample(
                    ctx_nchw,
                    src_xy_rot_gs,
                    mode="bilinear",
                    padding_mode=_pm,
                    align_corners=False,
                )
                warped_rot_nchw = self._apply_learned_oov(warped_rot_nchw, src_xy_rot_gs)
                warped_rot = warped_rot_nchw.permute(0, 2, 3, 1).reshape(B, HW, D)
                gate = depth_gate.to(dtype=dtype).reshape(B, HW, 1)
                warped[c] = gate * warped_proj + (1.0 - gate) * warped_rot
            else:
                warped[c] = warped_proj

        return torch.stack(warped, dim=1).reshape(B, T * HW, D)

    def _warp_last_context_to_target(
        self,
        x_raw: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        intrinsics: torch.Tensor,
        target_depth: torch.Tensor = None,
    ) -> torch.Tensor | None:
        """Projectively warp the last context frame's raw features to the
        prediction-target frame's grid.

        Used by the ``residual_target`` loss mode: the predictor learns
        ``z_target - warp(z_last, target)`` instead of the full ``z_target``,
        so the transformer only models what projective geometry cannot explain.

        Returns ``(B, HW, input_token_dim)`` or ``None`` when the warp is
        not applicable (T<2, warp_mode=="off", or missing inputs).
        """
        B, N_ctxt, D = x_raw.shape
        gh, gw = self.grid_height, self.grid_width
        HW = gh * gw
        T = N_ctxt // HW
        if T < 2 or self.warp_mode == "off":
            return None
        if states is None or actions is None or intrinsics is None:
            return None

        dtype = x_raw.dtype
        # ``states`` window = [ctx_0, ..., ctx_{T-2}, target].
        # Slot -2 is the last *context* frame; slot -1 is the target.
        # ``actions`` window has the same shape; slot -2 is the transition
        # from the last context frame into the target, and slot -1 is the
        # zero-padded outgoing action (no next frame to move to).
        src_state = states[:, -2].to(dtype=dtype)            # (B, 7) last context frame
        src_action = actions[:, -2].to(dtype=dtype)          # (B, 7) last-ctx → target
        tgt_state = self._compose_state_action(src_state, src_action)  # (B, 7)
        src_intr = intrinsics[:, -2].to(dtype=dtype)         # (B, 4)
        tgt_intr = intrinsics[:, -1].to(dtype=dtype)         # (B, 4) target frame K

        R_src = self._quat_xyzw_to_rotmat(src_state[..., 3:7])   # (B, 3, 3)
        R_tgt = self._quat_xyzw_to_rotmat(tgt_state[..., 3:7])   # (B, 3, 3)
        K_src = self._build_K_from_normalized(src_intr)          # (B, 3, 3)
        K_tgt_inv = self._build_K_inv_from_normalized(tgt_intr)  # (B, 3, 3)

        y_norm = (torch.arange(gh, device=x_raw.device, dtype=dtype) + 0.5) / gh
        x_norm = (torch.arange(gw, device=x_raw.device, dtype=dtype) + 0.5) / gw
        yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
        pixel_grid_tgt = torch.stack(
            [xx.reshape(-1), yy.reshape(-1), torch.ones(HW, device=x_raw.device, dtype=dtype)],
            dim=-1,
        )  # (HW, 3)

        if self.warp_mode in ("projective_probe", "projective_da3") and target_depth is not None:
            depth_t = self._prepare_target_depth(target_depth, B, HW, x_raw.device, dtype)
            rays_cam_tgt = torch.einsum("bij,nj->bni", K_tgt_inv, pixel_grid_tgt)
            pts_cam_tgt = rays_cam_tgt * depth_t.unsqueeze(-1)
            t_tgt_vec = tgt_state[..., :3].unsqueeze(1)
            pts_world = torch.einsum("bij,bnj->bni", R_tgt, pts_cam_tgt) + t_tgt_vec
            t_src_vec = src_state[..., :3].unsqueeze(1)
            pts_cam_src = torch.einsum("bji,bnj->bni", R_src, pts_world - t_src_vec)
            src_h = torch.einsum("bij,bnj->bni", K_src, pts_cam_src)
        else:
            R_st = torch.bmm(R_src.transpose(-1, -2), R_tgt)
            H_inv = torch.bmm(K_src, torch.bmm(R_st, K_tgt_inv))
            src_h = torch.einsum("bij,nj->bni", H_inv, pixel_grid_tgt)

        src_xy = src_h[..., :2] / src_h[..., 2:3].clamp_min(1.0e-6)
        src_xy_gs = (src_xy * 2.0 - 1.0).reshape(B, gh, gw, 2)

        src_features = x_raw[:, (T - 2) * HW:(T - 1) * HW, :]  # last context frame's raw features
        src_nchw = src_features.reshape(B, gh, gw, D).permute(0, 3, 1, 2).contiguous()
        _pm = "zeros" if self.warp_context_padding_mode == "learned" else self.warp_context_padding_mode
        warped_nchw = F.grid_sample(
            src_nchw,
            src_xy_gs,
            mode="bilinear",
            padding_mode=_pm,
            align_corners=False,
        )
        warped_nchw = self._apply_learned_oov(warped_nchw, src_xy_gs)
        return warped_nchw.permute(0, 2, 3, 1).reshape(B, HW, D)

    def build_target_canvas_inputs(
        self,
        context_tokens_raw: torch.Tensor,
        states: torch.Tensor,
        actions: torch.Tensor,
        intrinsics: torch.Tensor,
        target_depth: torch.Tensor = None,
    ) -> dict:
        B, _, D = context_tokens_raw.shape
        HW = self.grid_height * self.grid_width
        mask_tok = self.future_mask_token.to(
            device=context_tokens_raw.device,
            dtype=context_tokens_raw.dtype,
        ).expand(B, HW, -1)
        x_full = torch.cat([context_tokens_raw, mask_tok], dim=1)
        external_target_depth = target_depth
        target_depth = external_target_depth
        depth_gate = None
        if (
            self.warp_mode == "projective_probe"
            and self._depth_probe is not None
            and states is not None
            and intrinsics is not None
            and x_full.size(1) // HW > 1
        ):
            T_in = x_full.size(1) // HW
            x_target_raw = x_full[:, (T_in - 2) * HW:(T_in - 1) * HW, :]
            with torch.no_grad():
                if self._depth_probe_arch == "multiscale":
                    out = self._depth_probe(
                        x_target_raw,
                        grid_h=self.grid_height,
                        grid_w=self.grid_width,
                    )
                    if isinstance(out, dict):
                        log_depth = out["log_depth"]
                        log_sigma = out["log_sigma"]
                        sigma = torch.nn.functional.softplus(log_sigma) + 1.0e-3
                        depth_gate = torch.exp(-sigma)
                        if "visibility_logit" in out:
                            depth_gate = depth_gate * torch.sigmoid(out["visibility_logit"])
                    else:
                        log_depth = out
                else:
                    log_depth = self._depth_probe(x_target_raw).squeeze(-1)
                target_depth = (log_depth.exp() * self._depth_probe_alpha).clamp_min(1.0e-3)
        self._last_target_depth = target_depth.detach() if target_depth is not None else None
        warped = self._warp_last_context_to_target(
            x_full,
            states,
            actions,
            intrinsics,
            target_depth=target_depth,
        )
        if warped is None:
            warped = mask_tok
            oov = torch.ones(B, HW, 1, device=context_tokens_raw.device, dtype=context_tokens_raw.dtype)
        else:
            oov_raw = self._last_warp_oov_mask
            if oov_raw is None:
                oov = torch.zeros(B, HW, 1, device=context_tokens_raw.device, dtype=context_tokens_raw.dtype)
            else:
                oov = oov_raw.reshape(B, HW, 1).to(device=context_tokens_raw.device, dtype=context_tokens_raw.dtype)
        valid = 1.0 - oov
        confidence = valid
        return {
            "warped_context_raw": warped,
            "valid_mask": valid,
            "oov_mask": oov,
            "confidence_mask": confidence,
            "target_depth": target_depth,
        }

    def prepare_canvas_slot(
        self,
        warped_context_raw: torch.Tensor,
        valid_mask: torch.Tensor,
        oov_mask: torch.Tensor,
        boundary_mask: torch.Tensor = None,
        confidence_mask: torch.Tensor = None,
        rollout_step: int = 0,
        rot_weight: torch.Tensor = None,
    ) -> torch.Tensor:
        """Build the canvas target slot for canvas_first mode.

        Args:
            rot_weight: optional (B,) float tensor in [0, 1] used by
                canvas_warp_mode='rot_gated' to scale beta_valid per sample
                by the rotation dominance score.  Ignored for other modes.
        """
        B, HW, D = warped_context_raw.shape
        mask_tok = self.future_mask_token.to(
            device=warped_context_raw.device,
            dtype=warped_context_raw.dtype,
        ).expand(B, HW, -1)
        if boundary_mask is None:
            boundary_mask = warped_context_raw.new_zeros(B, HW, 1)
        else:
            boundary_mask = boundary_mask.to(device=warped_context_raw.device, dtype=warped_context_raw.dtype).reshape(B, HW, 1)
        valid_mask = valid_mask.to(device=warped_context_raw.device, dtype=warped_context_raw.dtype).reshape(B, HW, 1)
        oov_mask = oov_mask.to(device=warped_context_raw.device, dtype=warped_context_raw.dtype).reshape(B, HW, 1)
        if confidence_mask is None:
            confidence_mask = valid_mask
        else:
            confidence_mask = confidence_mask.to(device=warped_context_raw.device, dtype=warped_context_raw.dtype).reshape(B, HW, 1)
        type_ids = torch.zeros(B, HW, device=warped_context_raw.device, dtype=torch.long)
        type_ids = torch.where(boundary_mask.squeeze(-1) > 0.0, torch.ones_like(type_ids), type_ids)
        type_ids = torch.where(oov_mask.squeeze(-1) > 0.0, torch.full_like(type_ids, 2), type_ids)
        beta_values = self.canvas_beta_by_type.clamp(0.0, 1.0).to(
            device=warped_context_raw.device,
            dtype=warped_context_raw.dtype,
        )
        beta = beta_values[type_ids].unsqueeze(-1)
        if self.canvas_warp_mode == "rot_gated" and rot_weight is not None:
            # Scale beta_valid (type_id == 0) per sample by rotation dominance.
            # beta_boundary and beta_oov are kept as-is.
            rw = rot_weight.to(device=warped_context_raw.device, dtype=warped_context_raw.dtype)
            rw = rw.clamp(0.0, 1.0).reshape(B, 1, 1)
            valid_tok = (type_ids == 0).unsqueeze(-1)
            beta = torch.where(valid_tok, beta * rw, beta)
        if self.canvas_warp_mode == "projector":
            projected_delta = self.canvas_warp_proj(self.canvas_warp_norm(warped_context_raw))
            gamma = self.canvas_warp_gamma.to(device=warped_context_raw.device, dtype=warped_context_raw.dtype)
            canvas_warp = mask_tok + gamma * projected_delta
        elif self.canvas_warp_mode == "norm_clip":
            mask_norm_for_scale = mask_tok.norm(dim=-1, keepdim=True).mean().clamp_min(1.0e-6)
            warp_norm_for_scale = warped_context_raw.norm(dim=-1, keepdim=True).clamp_min(1.0e-6)
            canvas_warp = warped_context_raw * ((mask_norm_for_scale * self.canvas_norm_clip_ratio) / warp_norm_for_scale)
        else:
            canvas_warp = warped_context_raw
        target_canvas = beta * canvas_warp + (1.0 - beta) * mask_tok
        mask_embed = warped_context_raw.new_zeros(B, HW, D)
        type_embed = warped_context_raw.new_zeros(B, HW, D)
        step_embed = warped_context_raw.new_zeros(B, HW, D)
        if self.canvas_use_mask_feat_embed:
            mask_feats = torch.cat([valid_mask, oov_mask, boundary_mask, confidence_mask], dim=-1)
            mask_embed = self.canvas_mask_proj(mask_feats)
            target_canvas = target_canvas + mask_embed
        if self.canvas_use_token_type_embed:
            type_embed = self.canvas_type_embed(type_ids)
            target_canvas = target_canvas + type_embed
        if self.canvas_use_target_step_embed:
            step_id = int(max(0, min(int(rollout_step), self.canvas_target_step_embed.num_embeddings - 1)))
            step_ids = torch.full((B, HW), step_id, device=warped_context_raw.device, dtype=torch.long)
            step_embed = self.canvas_target_step_embed(step_ids)
            target_canvas = target_canvas + step_embed
        block_mask = self._sample_canvas_block_mask(valid_mask)
        if block_mask is not None:
            target_canvas = torch.where(block_mask, mask_tok, target_canvas)
        with torch.no_grad():
            mask_norm = mask_tok.norm(dim=-1).mean().clamp_min(1.0e-6)
            self._last_canvas_diag = {
                "valid_frac": float(valid_mask.mean().detach().item()),
                "oov_frac": float(oov_mask.mean().detach().item()),
                "boundary_frac": float(boundary_mask.mean().detach().item()),
                "beta_valid": float(beta_values[0].detach().item()),
                "beta_boundary": float(beta_values[1].detach().item()),
                "beta_oov": float(beta_values[2].detach().item()),
                "canvas_norm": float(target_canvas.norm(dim=-1).mean().detach().item()),
                "mask_token_norm": float(mask_norm.detach().item()),
                "warp_norm": float(warped_context_raw.norm(dim=-1).mean().detach().item()),
                "input_norm_ratio": float((warped_context_raw.norm(dim=-1).mean() / mask_norm).detach().item()),
                "canvas_warp_norm": float(canvas_warp.norm(dim=-1).mean().detach().item()),
                "canvas_warp_norm_ratio": float((canvas_warp.norm(dim=-1).mean() / mask_norm).detach().item()),
                "rot_weight_mean": float(rot_weight.mean().detach().item()) if rot_weight is not None else 1.0,
                "mask_embed_norm": float(mask_embed.norm(dim=-1).mean().detach().item()),
                "type_embed_norm": float(type_embed.norm(dim=-1).mean().detach().item()),
                "block_mask_frac": float(block_mask.to(dtype=valid_mask.dtype).mean().detach().item()) if block_mask is not None else 0.0,
            }
        return target_canvas

    def _sample_canvas_block_mask(self, valid_mask: torch.Tensor) -> torch.Tensor | None:
        if not (self.training and self.canvas_block_mask_enabled):
            return None
        B, HW, _ = valid_mask.shape
        gh, gw = self.grid_height, self.grid_width
        if HW != gh * gw:
            return None
        valid_bool = valid_mask.reshape(B, gh, gw) > 0.5
        if not bool(valid_bool.any().item()):
            return None
        mask = torch.zeros(B, gh, gw, device=valid_mask.device, dtype=torch.bool)
        min_area = max(1, int(round(float(HW) * self.canvas_block_mask_min_frac)))
        max_area = max(min_area, int(round(float(HW) * self.canvas_block_mask_max_frac)))
        for b in range(B):
            n_valid = int(valid_bool[b].sum().item())
            if n_valid <= 0:
                continue
            target_area = min(n_valid, int(torch.randint(min_area, max_area + 1, (1,), device=valid_mask.device).item()))
            n_rects = int(torch.randint(self.canvas_block_mask_min_rects, self.canvas_block_mask_max_rects + 1, (1,), device=valid_mask.device).item())
            for r in range(n_rects):
                remaining = target_area - int((mask[b] & valid_bool[b]).sum().item())
                if remaining <= 0:
                    break
                rect_area = max(1, remaining // max(1, n_rects - r))
                rect_h = int(torch.randint(1, gh + 1, (1,), device=valid_mask.device).item())
                rect_w = max(1, min(gw, int(math.ceil(rect_area / rect_h))))
                rect_h = max(1, min(gh, int(math.ceil(rect_area / rect_w))))
                y0 = int(torch.randint(0, gh - rect_h + 1, (1,), device=valid_mask.device).item())
                x0 = int(torch.randint(0, gw - rect_w + 1, (1,), device=valid_mask.device).item())
                mask[b, y0:y0 + rect_h, x0:x0 + rect_w] = True
            mask[b] &= valid_bool[b]
        if not bool(mask.any().item()):
            return None
        return mask.reshape(B, HW, 1)

    @staticmethod
    def _quat_xyzw_conjugate(q: torch.Tensor) -> torch.Tensor:
        return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)

    @staticmethod
    def _quat_xyzw_multiply(q1: torch.Tensor, q2: torch.Tensor) -> torch.Tensor:
        x1, y1, z1, w1 = q1[..., 0], q1[..., 1], q1[..., 2], q1[..., 3]
        x2, y2, z2, w2 = q2[..., 0], q2[..., 1], q2[..., 2], q2[..., 3]
        q = torch.stack([
            w1*x2 + x1*w2 + y1*z2 - z1*y2,
            w1*y2 - x1*z2 + y1*w2 + z1*x2,
            w1*z2 + x1*y2 - y1*x2 + z1*w2,
            w1*w2 - x1*x2 - y1*y2 - z1*z2,
        ], dim=-1)
        q = torch.nn.functional.normalize(q, dim=-1, eps=1.0e-8)
        sign = torch.where(q[..., 3:4] < 0.0, -1.0, 1.0)
        return q * sign

    @staticmethod
    def _context_to_target_actions_from_states(states: torch.Tensor) -> torch.Tensor:
        """Return direct relative actions from each slot to the target slot.

        The dataset action convention is ``state_t ∘ action_t = state_{t+1}``.
        Therefore the direct action from context slot ``i`` to target slot
        ``T-1`` is ``state_i^{-1} ∘ state_{T-1}``.
        """
        if states is None or states.size(-1) < 7:
            raise ValueError(
                "action_token_mode='target_relative' requires states with shape [B, T, >=7]."
            )
        t_ctx = states[..., :3]
        q_ctx = torch.nn.functional.normalize(states[..., 3:7], dim=-1, eps=1.0e-8)
        t_tgt = states[:, -1:, :3].expand_as(t_ctx)
        q_tgt = torch.nn.functional.normalize(
            states[:, -1:, 3:7].expand_as(q_ctx),
            dim=-1,
            eps=1.0e-8,
        )
        q_ctx_inv = CameraConditionedPredictorAC._quat_xyzw_conjugate(q_ctx)
        r_ctx_inv = CameraConditionedPredictorAC._quat_xyzw_to_rotmat(q_ctx_inv)
        t_rel = torch.einsum("...ij,...j->...i", r_ctx_inv, t_tgt - t_ctx)
        q_rel = CameraConditionedPredictorAC._quat_xyzw_multiply(q_ctx_inv, q_tgt)
        return torch.cat([t_rel, q_rel], dim=-1)

    def _action_tokens_for_mode(self, actions: torch.Tensor, states: torch.Tensor) -> torch.Tensor:
        if self.action_token_mode == "target_relative":
            return self._context_to_target_actions_from_states(states)
        return actions

    @staticmethod
    def _compose_state_action(state: torch.Tensor, action: torch.Tensor) -> torch.Tensor:
        """Compose absolute SE(3) state with a relative action.

        ``state`` and ``action`` are both 7D ``[tx, ty, tz, qx, qy, qz, qw]``
        (xyzw quaternion convention).  Returns ``state ∘ action``: the
        absolute pose reached by applying ``action`` after ``state``.
        """
        t_s, q_s = state[..., :3], state[..., 3:7]
        t_a, q_a = action[..., :3], action[..., 3:7]
        R_s = CameraConditionedPredictorAC._quat_xyzw_to_rotmat(q_s)
        t_new = t_s + torch.einsum("...ij,...j->...i", R_s, t_a)
        q_new = CameraConditionedPredictorAC._quat_xyzw_multiply(q_s, q_a)
        return torch.cat([t_new, q_new], dim=-1)

    @staticmethod
    def _quat_xyzw_to_rotmat(q: torch.Tensor) -> torch.Tensor:
        """Convert a (qx, qy, qz, qw) quaternion to a rotation matrix.

        Args:
            q: [..., 4] quaternion in xyzw order (matches the state layout
                produced by RE10KSequenceDataset).

        Returns:
            [..., 3, 3] rotation matrix that maps vectors from camera-local
            coordinates into the anchor-canonicalized frame.
        """
        q = torch.nn.functional.normalize(q, dim=-1, eps=1.0e-8)
        x, y, z, w = q.unbind(dim=-1)
        ww, xx, yy, zz = w * w, x * x, y * y, z * z
        wx, wy, wz = w * x, w * y, w * z
        xy, xz, yz = x * y, x * z, y * z
        row0 = torch.stack([ww + xx - yy - zz, 2.0 * (xy - wz),       2.0 * (xz + wy)],       dim=-1)
        row1 = torch.stack([2.0 * (xy + wz),   ww - xx + yy - zz,     2.0 * (yz - wx)],       dim=-1)
        row2 = torch.stack([2.0 * (xz - wy),   2.0 * (yz + wx),       ww - xx - yy + zz],     dim=-1)
        return torch.stack([row0, row1, row2], dim=-2)

    def _compute_ray_pe(self, intrinsics, states, B, T, D):
        """
        Compute per-patch Plücker-style ray embedding in the anchor frame.

        The intrinsics produce a per-patch direction in *camera-local*
        coordinates. Before concatenating with the translation (which is
        already in the anchor-canonicalized frame, see
        ``_canonicalize_states`` in train.py), the direction is rotated
        into the anchor frame by the per-frame quaternion so origin and
        direction live in a consistent coordinate system.

        Args:
            intrinsics: [B, T, 4] normalised [fx/W, fy/H, cx/W, cy/H].
            states: [B, T, 7] anchor-local pose [tx, ty, tz, qx, qy, qz, qw].

        Returns:
            [B, T, H*W, D] ray embeddings to add to patch tokens.
        """
        del B, D

        hw = self.grid_height * self.grid_width
        pixel_grid = self._ray_pixel_grid.to(device=intrinsics.device, dtype=intrinsics.dtype)
        px = pixel_grid[:, 0]
        py = pixel_grid[:, 1]

        fx = intrinsics[:, :, 0:1]
        fy = intrinsics[:, :, 1:2]
        cx = intrinsics[:, :, 2:3]
        cy = intrinsics[:, :, 3:4]

        ray_x = (px.unsqueeze(0).unsqueeze(0) - cx) / (fx + 1.0e-8)
        ray_y = (py.unsqueeze(0).unsqueeze(0) - cy) / (fy + 1.0e-8)
        ray_z = torch.ones_like(ray_x)

        ray_dir_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)                    # [B, T, HW, 3]
        ray_dir_cam = ray_dir_cam / (ray_dir_cam.norm(dim=-1, keepdim=True) + 1.0e-8)

        # Rotate camera-local direction into the anchor frame using the
        # per-frame quaternion.  The translation already lives in anchor
        # coordinates, so after this step (origin, direction) form a
        # geometrically consistent ray.
        R = self._quat_xyzw_to_rotmat(states[:, :, 3:7].to(dtype=ray_dir_cam.dtype))  # [B, T, 3, 3]
        ray_dir = torch.einsum("btij,bthj->bthi", R, ray_dir_cam)                   # [B, T, HW, 3]

        ray_origin = states[:, :, :3].to(dtype=ray_dir.dtype).unsqueeze(2).expand(-1, -1, hw, -1)

        # Dispatch on representation mode.
        #
        # ``origin_dir`` (legacy, default): raw concat of anchor-relative origin
        # and unit direction. Bit-identical to the original implementation.
        #
        # ``plucker``: canonical 6D Plücker line representation [d, m] with
        # m = o × d.  Invariant to the choice of point along the ray and
        # rotation-equivariant.  Same input dim (6) as legacy so the MLP
        # parameter shapes are unchanged; only the *meaning* of the 6D vector
        # differs.
        #
        # ``plucker_delta``: 12D extension that appends the per-frame delta
        # ``[d - d_anchor, m - m_anchor]`` at the same grid location, giving
        # the predictor an explicit "how did this token's ray move from
        # frame 0 to frame t" signal. d_anchor / m_anchor are the rays
        # for the same grid cell at frame 0 (states[:, 0, :]).
        if self.ray_pe_mode == "origin_dir":
            ray_feat = torch.cat([ray_origin, ray_dir], dim=-1)                     # [B, T, HW, 6]
        elif self.ray_pe_mode == "plucker":
            # Plücker moment m = o × d. Per-token cross product across the
            # last dim.  ``torch.cross`` works on broadcasted shapes when
            # dim=-1 is specified.
            ray_moment = torch.cross(ray_origin, ray_dir, dim=-1)                   # [B, T, HW, 3]
            ray_feat = torch.cat([ray_dir, ray_moment], dim=-1)                     # [B, T, HW, 6]
        elif self.ray_pe_mode == "plucker_delta":
            ray_moment = torch.cross(ray_origin, ray_dir, dim=-1)                   # [B, T, HW, 3]
            # Anchor (frame 0) ray at the same grid cell. Anchor-canonicalised
            # states put the anchor frame at identity, so its origin is zero
            # and its direction is the camera-local direction itself.
            R_anchor = R[:, 0:1]                                                    # [B, 1, 3, 3]
            ray_dir_anchor = torch.einsum(
                "bij,bhj->bhi", R_anchor[:, 0], ray_dir_cam[:, 0]
            ).unsqueeze(1).expand(-1, T, -1, -1)                                    # [B, T, HW, 3]
            ray_origin_anchor = (
                states[:, 0:1, :3].to(dtype=ray_dir.dtype).unsqueeze(2).expand(-1, T, hw, -1)
            )
            ray_moment_anchor = torch.cross(ray_origin_anchor, ray_dir_anchor, dim=-1)
            ray_feat = torch.cat(
                [ray_dir, ray_moment, ray_dir - ray_dir_anchor, ray_moment - ray_moment_anchor],
                dim=-1,
            )                                                                       # [B, T, HW, 12]
        elif self.ray_pe_mode == "plucker_pair":
            # RayMap v4c — pair each frame's ray with the previous frame's
            # ray at the same grid cell. Per-token features give the
            # predictor an explicit per-token camera-displacement signal and
            # (optionally) a rotation-only warp correspondence into the
            # previous frame so the predictor can distinguish visible vs
            # disoccluded / border tokens.
            ray_moment = torch.cross(ray_origin, ray_dir, dim=-1)                   # [B, T, HW, 3]
            # Previous-frame ray at the same grid cell. For t=0 we pair the
            # frame with itself (Δ=0), which keeps shapes consistent without
            # contaminating any interior frame.
            ray_dir_prev = torch.cat(
                [ray_dir[:, 0:1], ray_dir[:, :-1]], dim=1
            )                                                                       # [B, T, HW, 3]
            ray_origin_prev = torch.cat(
                [ray_origin[:, 0:1], ray_origin[:, :-1]], dim=1
            )                                                                       # [B, T, HW, 3]
            ray_moment_prev = torch.cross(ray_origin_prev, ray_dir_prev, dim=-1)    # [B, T, HW, 3]
            delta_o = ray_origin - ray_origin_prev                                  # [B, T, HW, 3]
            delta_d = ray_dir - ray_dir_prev                                        # [B, T, HW, 3]
            dot_dd = (ray_dir * ray_dir_prev).sum(dim=-1, keepdim=True)             # [B, T, HW, 1]
            cross_dd = torch.cross(ray_dir, ray_dir_prev, dim=-1)                   # [B, T, HW, 3]
            pair_feats = [
                ray_dir, ray_moment,
                ray_dir_prev, ray_moment_prev,
                delta_o, delta_d,
                dot_dd, cross_dd,
            ]

            if self.ray_visibility_features == "warp_plus_border":
                # Rotation-only warp of the target patch (frame t, pixel
                # uv) into the previous frame (frame t-1). Uses the anchor-
                # frame world direction ``ray_dir`` (already R_t d_cam),
                # rotates it into frame t-1's camera by R_{t-1}^T, then
                # projects with K_{t-1}. Translation is ignored by design
                # (rotation-only homography = infinity-plane warp). For
                # RE10K small-baseline clips this is a well-defined cheap
                # correspondence field; correct for pure rotation and
                # well-behaved for moderate translation.
                R_prev = torch.cat([R[:, 0:1], R[:, :-1]], dim=1)                   # [B, T, 3, 3]
                d_in_prev_cam = torch.einsum(
                    "btji,bthj->bthi", R_prev, ray_dir
                )                                                                    # R_prev^T · ray_dir
                d_z = d_in_prev_cam[..., 2:3]
                eps = 1.0e-6
                # Project using frame (t-1) intrinsics.
                fx_prev = torch.cat([fx[:, 0:1], fx[:, :-1]], dim=1)                 # [B, T, 1]
                fy_prev = torch.cat([fy[:, 0:1], fy[:, :-1]], dim=1)
                cx_prev = torch.cat([cx[:, 0:1], cx[:, :-1]], dim=1)
                cy_prev = torch.cat([cy[:, 0:1], cy[:, :-1]], dim=1)
                fx_prev = fx_prev.unsqueeze(2)                                       # [B, T, 1, 1]
                fy_prev = fy_prev.unsqueeze(2)
                cx_prev = cx_prev.unsqueeze(2)
                cy_prev = cy_prev.unsqueeze(2)
                safe_z = torch.where(d_z.abs() > eps, d_z, torch.full_like(d_z, eps))
                uv_x = fx_prev * (d_in_prev_cam[..., 0:1] / safe_z) + cx_prev        # [B, T, HW, 1]
                uv_y = fy_prev * (d_in_prev_cam[..., 1:2] / safe_z) + cy_prev        # [B, T, HW, 1]
                valid = (
                    (d_z > eps)
                    & (uv_x >= 0.0) & (uv_x <= 1.0)
                    & (uv_y >= 0.0) & (uv_y <= 1.0)
                ).to(ray_dir.dtype)                                                  # [B, T, HW, 1]
                # Signed border distance: positive inside, negative outside,
                # clamped to [-0.5, 0.5] so gradients remain bounded when
                # the warped pixel lands far outside the frame.
                border = torch.minimum(
                    torch.minimum(uv_x, 1.0 - uv_x),
                    torch.minimum(uv_y, 1.0 - uv_y),
                ).clamp(-0.5, 0.5)                                                   # [B, T, HW, 1]
                # Neutralise visibility features at t=0 (self-pair): warp
                # coords land on native pixel grid with valid=1 by
                # construction, which is consistent with "previous=self"
                # semantics elsewhere in this mode. No special-case needed.
                pair_feats.extend([uv_x, uv_y, valid, border])

            ray_feat = torch.cat(pair_feats, dim=-1)                                 # [B, T, HW, 22|26]
        else:  # pragma: no cover -- guarded in __init__
            raise ValueError(f"ray_pe_mode={self.ray_pe_mode!r} unsupported.")

        return self.ray_pe_mlp(ray_feat)

    def _compute_action_ray_pe(self, actions, intrinsics, B, T, D):
        """Per-patch dense encoding of the relative-pose action.

        Mirrors ``_compute_ray_pe`` but the rotation/translation come from
        ``actions`` (relative pose) rather than ``states`` (absolute pose).
        The per-patch camera-local direction ``d_cam[u]`` is rotated by
        ``R_action`` to give the post-action direction in the current
        camera frame, and paired with the action translation as the ray
        origin. The Plücker line ``[d, o×d]`` is fed through a separate
        MLP (``action_ray_pe_mlp``) so state and action encodings learn
        independent feature spaces.

        Args:
            actions: [B, T, 7] relative pose [tx, ty, tz, qx, qy, qz, qw].
            intrinsics: [B, T, 4] normalised [fx/W, fy/H, cx/W, cy/H].

        Returns:
            [B, T, H*W, D] action-ray embedding to add to patch tokens.
        """
        del B, D

        hw = self.grid_height * self.grid_width
        pixel_grid = self._ray_pixel_grid.to(device=intrinsics.device, dtype=intrinsics.dtype)
        px = pixel_grid[:, 0]
        py = pixel_grid[:, 1]

        fx = intrinsics[:, :, 0:1]
        fy = intrinsics[:, :, 1:2]
        cx = intrinsics[:, :, 2:3]
        cy = intrinsics[:, :, 3:4]

        ray_x = (px.unsqueeze(0).unsqueeze(0) - cx) / (fx + 1.0e-8)
        ray_y = (py.unsqueeze(0).unsqueeze(0) - cy) / (fy + 1.0e-8)
        ray_z = torch.ones_like(ray_x)
        ray_dir_cam = torch.stack([ray_x, ray_y, ray_z], dim=-1)                    # [B, T, HW, 3]
        ray_dir_cam = ray_dir_cam / (ray_dir_cam.norm(dim=-1, keepdim=True) + 1.0e-8)

        # Rotate per-patch camera-local direction by the action rotation.
        R_action = self._quat_xyzw_to_rotmat(
            actions[:, :, 3:7].to(dtype=ray_dir_cam.dtype)
        )                                                                             # [B, T, 3, 3]
        ray_dir_action = torch.einsum("btij,bthj->bthi", R_action, ray_dir_cam)      # [B, T, HW, 3]

        # Origin = action translation, broadcast across patches.
        ray_origin_action = actions[:, :, :3].to(dtype=ray_dir_action.dtype)         # [B, T, 3]
        ray_origin_action = ray_origin_action.unsqueeze(2).expand(-1, -1, hw, -1)    # [B, T, HW, 3]

        # Plücker line: m = o × d.
        ray_moment_action = torch.cross(ray_origin_action, ray_dir_action, dim=-1)   # [B, T, HW, 3]
        action_feat = torch.cat([ray_dir_action, ray_moment_action], dim=-1)         # [B, T, HW, 6]

        return self.action_ray_pe_mlp(action_feat)

    def _compute_coord_qk_embed(
        self,
        geometry_depths: torch.Tensor,
        states: torch.Tensor,
        intrinsics: torch.Tensor,
        B: int,
        T: int,
        D: int,
    ) -> torch.Tensor:
        """Project depth-aware token coordinates into the final target camera."""
        del D
        if geometry_depths.ndim != 4:
            raise ValueError(
                f"geometry_depths must be (B, T, H, W); got {tuple(geometry_depths.shape)}"
            )
        if geometry_depths.shape[0] != B or geometry_depths.shape[1] != T:
            raise ValueError(
                f"geometry_depths shape {tuple(geometry_depths.shape)} incompatible with B={B}, T={T}"
            )
        HW = self.grid_height * self.grid_width
        H_img = int(geometry_depths.shape[-2])
        W_img = int(geometry_depths.shape[-1])
        dtype = intrinsics.dtype
        device = intrinsics.device

        depth_tok = F.avg_pool2d(
            geometry_depths.reshape(B * T, 1, H_img, W_img).to(device=device, dtype=torch.float32),
            kernel_size=self.patch_size,
            stride=self.patch_size,
        ).reshape(B, T, HW).to(dtype=dtype).clamp_min(1.0e-3)

        pixel_grid = self._ray_pixel_grid.to(device=device, dtype=dtype)
        px = pixel_grid[:, 0]
        py = pixel_grid[:, 1]
        fx = intrinsics[:, :, 0:1].to(dtype=dtype).clamp_min(1.0e-6)
        fy = intrinsics[:, :, 1:2].to(dtype=dtype).clamp_min(1.0e-6)
        cx = intrinsics[:, :, 2:3].to(dtype=dtype)
        cy = intrinsics[:, :, 3:4].to(dtype=dtype)

        x_cam = (px.view(1, 1, HW) - cx) / fx * depth_tok
        y_cam = (py.view(1, 1, HW) - cy) / fy * depth_tok
        z_cam = depth_tok
        pts_cam = torch.stack([x_cam, y_cam, z_cam], dim=-1)

        R_src = self._quat_xyzw_to_rotmat(states[:, :, 3:7].to(dtype=dtype))
        t_src = states[:, :, :3].to(dtype=dtype).unsqueeze(2)
        pts_world = torch.einsum("btij,bthj->bthi", R_src, pts_cam) + t_src

        R_tgt = R_src[:, -1]
        t_tgt = states[:, -1, :3].to(dtype=dtype).view(B, 1, 1, 3)
        pts_tgt = torch.einsum("bij,bthj->bthi", R_tgt.transpose(-1, -2), pts_world - t_tgt)
        z_tgt = pts_tgt[..., 2:3].clamp_min(1.0e-3)

        fx_t = intrinsics[:, -1, 0:1].to(dtype=dtype).view(B, 1, 1)
        fy_t = intrinsics[:, -1, 1:2].to(dtype=dtype).view(B, 1, 1)
        cx_t = intrinsics[:, -1, 2:3].to(dtype=dtype).view(B, 1, 1)
        cy_t = intrinsics[:, -1, 3:4].to(dtype=dtype).view(B, 1, 1)
        u_t = fx_t * (pts_tgt[..., 0] / z_tgt[..., 0]) + cx_t
        v_t = fy_t * (pts_tgt[..., 1] / z_tgt[..., 0]) + cy_t
        log_z = torch.tanh(torch.log(z_tgt[..., 0].clamp_min(1.0e-3)) / 4.0)

        coord = torch.stack(
            [(u_t - 0.5) * 2.0, (v_t - 0.5) * 2.0, log_z],
            dim=-1,
        ).clamp(-4.0, 4.0)
        freqs = torch.arange(self.coord_qk_freqs, device=device, dtype=dtype)
        freqs = (2.0 ** freqs).view(1, 1, 1, 1, self.coord_qk_freqs)
        phase = coord.unsqueeze(-1) * freqs * math.pi
        feat = torch.cat(
            [coord, torch.sin(phase).flatten(-2), torch.cos(phase).flatten(-2)],
            dim=-1,
        )
        mlp_dtype = next(self.coord_qk_mlp.parameters()).dtype
        return self.coord_qk_mlp(feat.to(dtype=mlp_dtype)).to(dtype=dtype)

    def _compute_target_motion_query(
        self,
        states: torch.Tensor,
        intrinsics: torch.Tensor,
        B: int,
        T: int,
        D: int,
    ) -> torch.Tensor:
        """Projected same-pixel displacement query from frame t to frame t-1.

        This is a WorldCam-style explicit camera-motion side channel. For each
        patch center in frame t, it backprojects a unit-depth point, transforms
        it into the previous camera, projects it, and encodes the resulting
        optical displacement plus validity/depth and relative translation.
        Unit depth avoids target-frame depth leakage while still exposing the
        direction and approximate magnitude of camera-induced parallax.
        """
        del D
        HW = self.grid_height * self.grid_width
        dtype = intrinsics.dtype
        device = intrinsics.device
        pixel_grid = self._ray_pixel_grid.to(device=device, dtype=dtype)
        px = pixel_grid[:, 0]
        py = pixel_grid[:, 1]

        fx_t = intrinsics[:, :, 0:1].to(dtype=dtype).clamp_min(1.0e-6)
        fy_t = intrinsics[:, :, 1:2].to(dtype=dtype).clamp_min(1.0e-6)
        cx_t = intrinsics[:, :, 2:3].to(dtype=dtype)
        cy_t = intrinsics[:, :, 3:4].to(dtype=dtype)
        ray_x = (px.view(1, 1, HW) - cx_t) / fx_t
        ray_y = (py.view(1, 1, HW) - cy_t) / fy_t
        pts_cam_t = torch.stack([ray_x, ray_y, torch.ones_like(ray_x)], dim=-1)

        R_t = self._quat_xyzw_to_rotmat(states[:, :, 3:7].to(dtype=dtype))
        t_t = states[:, :, :3].to(dtype=dtype).unsqueeze(2)
        pts_world = torch.einsum("btij,bthj->bthi", R_t, pts_cam_t) + t_t

        states_prev = torch.cat([states[:, 0:1], states[:, :-1]], dim=1).to(dtype=dtype)
        intr_prev = torch.cat([intrinsics[:, 0:1], intrinsics[:, :-1]], dim=1).to(dtype=dtype)
        R_prev = self._quat_xyzw_to_rotmat(states_prev[:, :, 3:7])
        t_prev = states_prev[:, :, :3].unsqueeze(2)
        pts_prev = torch.einsum("btji,bthj->bthi", R_prev, pts_world - t_prev)
        z_prev = pts_prev[..., 2:3]

        fx_p = intr_prev[:, :, 0:1].clamp_min(1.0e-6).unsqueeze(2)
        fy_p = intr_prev[:, :, 1:2].clamp_min(1.0e-6).unsqueeze(2)
        cx_p = intr_prev[:, :, 2:3].unsqueeze(2)
        cy_p = intr_prev[:, :, 3:4].unsqueeze(2)
        safe_z = torch.where(z_prev.abs() > 1.0e-6, z_prev, torch.full_like(z_prev, 1.0e-6))
        uv_prev_x = fx_p * (pts_prev[..., 0:1] / safe_z) + cx_p
        uv_prev_y = fy_p * (pts_prev[..., 1:2] / safe_z) + cy_p
        uv_prev = torch.cat([uv_prev_x, uv_prev_y], dim=-1)
        uv_cur = pixel_grid.view(1, 1, HW, 2).expand(B, T, -1, -1)
        duv = (uv_cur - uv_prev).clamp(-4.0, 4.0)
        valid = (
            (z_prev > 1.0e-6)
            & (uv_prev_x >= 0.0) & (uv_prev_x <= 1.0)
            & (uv_prev_y >= 0.0) & (uv_prev_y <= 1.0)
        ).to(dtype=dtype)
        border = torch.minimum(
            torch.minimum(uv_prev_x, 1.0 - uv_prev_x),
            torch.minimum(uv_prev_y, 1.0 - uv_prev_y),
        ).clamp(-0.5, 0.5)
        t_rel_prev = torch.einsum(
            "btji,btj->bti",
            R_prev,
            states[:, :, :3].to(dtype=dtype) - states_prev[:, :, :3],
        )
        t_rel_prev[:, 0].zero_()
        t_rel_tok = t_rel_prev.unsqueeze(2).expand(-1, -1, HW, -1)
        t_norm = t_rel_prev.norm(dim=-1, keepdim=True).unsqueeze(2).expand(-1, -1, HW, -1)
        log_z = torch.tanh(torch.log(z_prev.abs().clamp_min(1.0e-3)) / 4.0)
        feat = torch.cat(
            [
                uv_prev.clamp(-4.0, 4.0),
                duv,
                valid,
                border,
                log_z,
                t_rel_tok.clamp(-4.0, 4.0),
                t_norm.clamp(0.0, 4.0),
            ],
            dim=-1,
        )
        mlp_dtype = next(self.target_motion_query_mlp.parameters()).dtype
        return self.target_motion_query_mlp(feat.to(dtype=mlp_dtype)).to(dtype=dtype)

    @staticmethod
    def _interleave_visual_sidechannel(visual_embed: torch.Tensor, cond_tokens: int) -> torch.Tensor:
        B, T, _HW, D = visual_embed.shape
        if cond_tokens <= 0:
            return visual_embed.flatten(1, 2)
        zeros = visual_embed.new_zeros(B, T, cond_tokens, D)
        return torch.cat([zeros, visual_embed], dim=2).flatten(1, 2)

    def _build_adaln_condition(
        self,
        action_token_embed: torch.Tensor,
        state_token_embed: torch.Tensor,
        intrinsics_token_embed: torch.Tensor | None,
    ) -> torch.Tensor | None:
        if not self.predictor_adaln_enabled:
            return None
        pooled = [
            action_token_embed.squeeze(2).mean(dim=1),
            state_token_embed.squeeze(2).mean(dim=1),
        ]
        if self.use_intrinsics:
            if intrinsics_token_embed is None:
                raise RuntimeError("AdaLN+RoPE predictor expected intrinsics conditioning.")
            pooled.append(intrinsics_token_embed.squeeze(2).mean(dim=1))
        cond = torch.cat(pooled, dim=-1)
        cond_dtype = next(self.predictor_adaln_conditioner.parameters()).dtype
        return self.predictor_adaln_conditioner(cond.to(dtype=cond_dtype)).to(dtype=action_token_embed.dtype)

    def _forward_adaln_block(
        self,
        blk: nn.Module,
        block_idx: int,
        x: torch.Tensor,
        adaln_cond: torch.Tensor,
        mask=None,
        attn_mask=None,
        T=None,
        H=None,
        W=None,
        action_tokens=0,
        qk_mod=None,
    ) -> torch.Tensor:
        sh_a, sc_a, ga_a, sh_m, sc_m, ga_m = self.predictor_adaln_modulators[block_idx](adaln_cond)

        y = _adaln_modulate(blk.norm1(x), sh_a, sc_a)
        if isinstance(blk.attn, ACRoPEAttention):
            y = blk.attn(
                y,
                mask=mask,
                attn_mask=attn_mask,
                T=T,
                H=H,
                W=W,
                action_tokens=action_tokens,
                qk_mod=qk_mod,
            )
        else:
            y = blk.attn(y, mask=mask, attn_mask=attn_mask)
        x = x + blk.drop_path((1.0 + ga_a.unsqueeze(1)) * y)

        y = _adaln_modulate(blk.norm2(x), sh_m, sc_m)
        x = x + blk.drop_path((1.0 + ga_m.unsqueeze(1)) * blk.mlp(y))
        return x

    # ---------------------------------------------------------------------- #
    # Forward
    # ---------------------------------------------------------------------- #

    def forward(self, x, actions, states, intrinsics=None, target_depth=None, geometry_depths=None):
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
            geometry_depths: Optional projective depth for all frame slots,
               shape [B, T, H, W]. When coord_qk is enabled this builds
               depth-aware coordinate embeddings added to attention Q/K only.

        Returns:
            predictions: Projected frame-level predictions.
               Shape: [B, T * H * W, n_hierarchical_layers * out_embed_dim].
            context_predictions: Projected context-token predictions (only if
               predict_all=True, else None).
               Shape: [B, T * H * W, n_hierarchical_layers * out_embed_dim].
            delta_predictions: Projected delta predictions (only if
               use_delta_head=True, else None).
               Shape: [B, T * H * W, n_hierarchical_layers * out_embed_dim].
            fsq_logits: Categorical logits over FSQ code axes (only if
               use_fsq_head=True, else None).
               Shape: [B, T * H * W, fsq_total_code_axes, fsq_levels].
            warped_context_raw: Last context frame projectively warped to the
               target frame's grid, in raw encoder space (only if warp_mode
               != "off" and T>1, else None).  Used by ``residual_target`` loss.
               Shape: [B, HW, n_hierarchical_layers * embed_dim].
        """
        capture_ray_debug = bool(getattr(self, "_capture_ray_conditioning", False))
        self._last_ray_conditioning = None
        ray_debug = {} if capture_ray_debug else None

        # Path B-lite Step 1: optionally compute target-frame depth from
        # the **raw** 4-layer concat features (pre ``predictor_embed``).
        # The probe is a frozen ``nn.Linear(input_token_dim, 1)`` distilled
        # from DA3 with a single global α scalar baked into the checkpoint.
        # Output ``target_depth`` has shape ``(B, HW)`` and is in the same
        # pose-consistent units that ``_warp_context_tokens`` expects.
        external_target_depth = target_depth
        target_depth = external_target_depth
        depth_gate = None
        if (
            self.warp_mode == "projective_probe"
            and self._depth_probe is not None
            and states is not None
            and intrinsics is not None
        ):
            HW_full = self.grid_height * self.grid_width
            T_in = x.size(1) // HW_full
            if T_in > 1:
                # Probe the *last context frame's* raw encoder features (the
                # second-to-last slot) to estimate source-frame depth.  The
                # final slot ``x[:, -HW_full:]`` is the future_mask_token — a
                # learned constant that carries zero scene information, so
                # running the depth probe there produces scene-invariant
                # (position-only) depth and makes the projective warp useless.
                # Using the last *real* context features gives the probe access
                # to the actual scene geometry adjacent to the target frame.
                x_target_raw = x[:, (T_in - 2) * HW_full:(T_in - 1) * HW_full, :]  # (B, HW, input_token_dim)
                with torch.no_grad():
                    if self._depth_probe_arch == "multiscale":
                        out = self._depth_probe(
                            x_target_raw,
                            grid_h=self.grid_height,
                            grid_w=self.grid_width,
                        )
                        if isinstance(out, dict):
                            log_depth = out["log_depth"]
                            # v1.5 gate: g(u) = exp(-σ(u)) ∈ (0, 1].
                            # High-σ tokens get g near 0 → fall back to
                            # rotation-only warp; low-σ tokens get g near
                            # 1 → use the projective warp. Optional
                            # multiplicative visibility mask (sigmoid)
                            # zero-pads disoccluded tokens.
                            log_sigma = out["log_sigma"]
                            sigma = torch.nn.functional.softplus(log_sigma) + 1.0e-3
                            depth_gate = torch.exp(-sigma)
                            if "visibility_logit" in out:
                                vis = torch.sigmoid(out["visibility_logit"])
                                depth_gate = depth_gate * vis
                        else:
                            log_depth = out                              # (B, HW)
                    else:
                        log_depth = self._depth_probe(x_target_raw).squeeze(-1)  # (B, HW)
                    target_depth = (
                        log_depth.exp() * self._depth_probe_alpha
                    ).clamp_min(1.0e-3)

        # -- project visual tokens to predictor dimension
        x_raw = x  # save raw encoder features for residual_target warp
        x = self.predictor_embed(x)
        B, N_ctxt, D = x.size()
        T = N_ctxt // (self.grid_height * self.grid_width)
        HW = self.grid_height * self.grid_width
        if ray_debug is not None:
            ray_debug["raw_pre_embed_target"] = x_raw.view(B, T, HW, -1)[:, -1].detach()
            ray_debug["post_embed_target"] = x.view(B, T, HW, D)[:, -1].detach()

        if self.use_intrinsics and intrinsics is None:
            raise ValueError(
                "CameraConditionedPredictorAC was built with use_intrinsics=True "
                "but forward() received intrinsics=None. The attention mask and "
                "post-block reshape assume cond_tokens=3; passing None would "
                "silently drop the K slot and break the sequence layout."
            )

        # Optional: pixel-align context frames to the target frame.  Mode
        # is selected by ``self.warp_mode``: ``"rotation_only"`` uses the
        # legacy infinity-plane homography; ``"projective_probe"`` uses
        # the depth-probe target depth computed above.  No-op when
        # ``warp_mode == "off"``, T<2, or the required pose/K inputs are
        # missing.
        if (
            self.warp_mode != "off"
            and T > 1
            and states is not None
            and intrinsics is not None
        ):
            x = self._warp_context_tokens(
                x, states, intrinsics,
                target_depth=target_depth,
                depth_gate=depth_gate,
            )
            if ray_debug is not None:
                ray_debug["post_warp_target"] = x.view(B, T, HW, D)[:, -1].detach()

        # P0.6: expose target_depth for diagnostic logging in train.py without
        # changing the return signature.  Reset to None at the start of each
        # forward to prevent stale values / retained graph references.
        self._last_target_depth = None
        self._last_target_depth = target_depth.detach() if target_depth is not None else None

        # Compute warped last-context-frame features in raw encoder space
        # for residual_target loss mode (Run B).
        warped_context_raw = self._warp_last_context_to_target(
            x_raw, states, actions, intrinsics, target_depth=target_depth
        )

        ray_embed = None
        if self.use_ray_pe and intrinsics is not None:
            ray_embed = self._compute_ray_pe(intrinsics, states, B, T, D)
            x_with_ray = x.view(B, T, HW, D)
            if ray_debug is not None:
                ray_debug["pre_ray_target"] = x_with_ray[:, -1].detach()
                ray_debug["ray_pe_target"] = ray_embed[:, -1].detach()
            x = (x_with_ray + ray_embed).flatten(1, 2)
            if ray_debug is not None:
                ray_debug["post_ray_target"] = x.view(B, T, HW, D)[:, -1].detach()

        # raymap_dual: also add a dense per-patch action raymap so the action
        # signal is delivered through the same per-token pathway as state,
        # rather than via the sparse ``action_encoder`` token.
        if (
            self.pose_conditioning_mode == "raymap_dual"
            and self.use_ray_pe
            and intrinsics is not None
        ):
            action_embed = self._compute_action_ray_pe(actions, intrinsics, B, T, D)
            x_with_act = x.view(B, T, HW, D)
            if ray_debug is not None:
                ray_debug["action_ray_pe_target"] = action_embed[:, -1].detach()
            x = (x_with_act + action_embed).flatten(1, 2)
            if ray_debug is not None:
                ray_debug["post_action_ray_target"] = x.view(B, T, HW, D)[:, -1].detach()

        if self.target_motion_query_enabled and states is not None and intrinsics is not None:
            motion_query = self._compute_target_motion_query(states, intrinsics, B, T, D)
            gamma = self.target_motion_query_gamma.to(device=x.device, dtype=x.dtype)
            x_with_motion = x.view(B, T, HW, D)
            x = (x_with_motion + gamma * motion_query.to(dtype=x.dtype)).flatten(1, 2)
            if ray_debug is not None:
                ray_debug["post_motion_query_target"] = x.view(B, T, HW, D)[:, -1].detach()

        # -- encode camera conditioning tokens → [B, T, 1, D] each.
        # ``transition`` preserves the original dataset delta token. The
        # target-relative mode instead gives every context slot its direct SE(3)
        # action to the target frame, which is better aligned with rollout
        # prediction than the local previous→current transition.
        action_tokens = self._action_tokens_for_mode(actions, states)
        a = self.action_encoder(action_tokens).unsqueeze(2)    # [B, T, 1, D]
        s = self.state_encoder(states).unsqueeze(2)             # [B, T, 1, D]
        k = None
        if self.use_intrinsics and intrinsics is not None:
            k = self.intrinsics_encoder(intrinsics).unsqueeze(2)
        adaln_cond = self._build_adaln_condition(a, s, k)
        if ray_debug is not None:
            ray_debug["action_token_target"] = a[:, -1, 0].detach()
            ray_debug["state_token_target"] = s[:, -1, 0].detach()
            if k is not None:
                ray_debug["intrinsics_token_target"] = k[:, -1, 0].detach()

        # -- reshape visual tokens for per-frame interleaving
        x_frames = x.view(B, T, HW, D)  # [B, T, H*W, D]

        # Pose-conditioning layout (see ``self.pose_conditioning_mode`` in
        # ``__init__``). ``token+raymap`` (default, legacy): include the dense
        # ``state_encoder(states)`` token. ``raymap_only`` (DA3 RayMap v3):
        # drop the state token; pose information lives entirely in the
        # per-patch ray PE that was already added to ``x_frames`` above. The
        # resulting ``cond_tokens`` value (set in ``__init__``) MUST match
        # the per-frame conditioning layout below or the precomputed
        # ``attn_mask`` and post-block reshape will silently mis-slice.
        if self.pose_conditioning_mode == "token+raymap":
            if self.use_intrinsics and intrinsics is not None:
                # interleave: [action | state | intrinsics | patches]
                x_seq = torch.cat([a, s, k, x_frames], dim=2).flatten(1, 2)  # [B, T*(3+H*W), D]
            else:
                # interleave: [action | state | patches]
                x_seq = torch.cat([a, s, x_frames], dim=2).flatten(1, 2)     # [B, T*(2+H*W), D]
        elif self.pose_conditioning_mode == "raymap_only":
            if self.use_intrinsics and intrinsics is not None:
                # interleave: [action | intrinsics | patches+raymap]
                x_seq = torch.cat([a, k, x_frames], dim=2).flatten(1, 2)     # [B, T*(2+H*W), D]
            else:
                # interleave: [action | patches+raymap]
                x_seq = torch.cat([a, x_frames], dim=2).flatten(1, 2)        # [B, T*(1+H*W), D]
        else:  # pose_conditioning_mode == "raymap_dual"
            # Both sparse state and sparse action tokens are dropped. State
            # information lives in the per-patch state raymap; action
            # information lives in the per-patch action raymap (both already
            # added to ``x_frames`` above).
            if self.use_intrinsics and intrinsics is not None:
                # interleave: [intrinsics | patches+state_raymap+action_raymap]
                x_seq = torch.cat([k, x_frames], dim=2).flatten(1, 2)        # [B, T*(1+H*W), D]
            else:
                # interleave: [patches+state_raymap+action_raymap]
                x_seq = x_frames.flatten(1, 2)                                # [B, T*(H*W), D]
        if ray_debug is not None:
            x_seq_frames_debug = x_seq.view(B, T, self.cond_tokens + HW, D)
            ray_debug["after_concat_target_cond"] = x_seq_frames_debug[:, -1, :self.cond_tokens, :].detach()
            ray_debug["after_concat_target_visual"] = x_seq_frames_debug[:, -1, self.cond_tokens:, :].detach()

        coord_qk_seq = None
        coord_qk_apply_idx_map = {}
        ray_qk_seq = None
        ray_qk_apply_idx_map = {}
        if self.coord_qk_enabled and states is not None and intrinsics is not None:
            if geometry_depths is None:
                geometry_depths = x_seq.new_ones(B, T, self.img_height, self.img_width)
            coord_visual = self._compute_coord_qk_embed(
                geometry_depths,
                states,
                intrinsics,
                B,
                T,
                D,
            )
            coord_qk_seq = self._interleave_visual_sidechannel(coord_visual, self.cond_tokens)
            coord_qk_apply_idx_map = {
                block_idx: pos
                for pos, block_idx in enumerate(self.coord_qk_apply_layers)
            }
        if self.ray_qk_enabled:
            if ray_embed is None:
                raise RuntimeError("ray_qk_enabled=True requires ray_embed in forward().")
            ray_qk_seq = self._interleave_visual_sidechannel(ray_embed, self.cond_tokens)
            ray_qk_apply_idx_map = {
                block_idx: pos
                for pos, block_idx in enumerate(self.ray_qk_apply_layers)
            }

        # -- slice causal mask to actual sequence length
        seq_len = x_seq.size(1)
        attn_mask = None
        if self.attn_mask is not None:
            attn_mask = self.attn_mask[:seq_len, :seq_len].to(x_seq.device, non_blocking=True).clone()

        # Phase E1 — rotation-homography correspondence bias. Computed once
        # per forward (depends only on per-frame poses + intrinsics + token
        # grid), then mixed into each applicable block's ``attn_mask`` via a
        # per-layer learnable lambda multiplied by the per-epoch ramp.
        corr_bias_base = None         # (B, seq_len, seq_len), shared across layers
        corr_lambda_eff = None        # (n_apply,) effective lambda per applied layer
        corr_apply_idx_map = {}       # block_idx -> position in correspondence_bias_lambda
        if (
            self.correspondence_bias_enabled
            and T > 1
            and states is not None
            and intrinsics is not None
        ):
            ramp = float(self.correspondence_bias_ramp.item())
            corr_lambda_eff_candidate = self.correspondence_bias_lambda * ramp
            # Short-circuit when every effective lambda is exactly zero
            # (e.g. lambda_init=0 before any optimiser step, or ramp=0).
            # Skipping the bias compute + (B, 1, N, N) mask reshape preserves
            # the original (N, N) attn_mask shape, which keeps SDPA on the
            # FLASH / EFFICIENT kernels and avoids backend-induced ~1e-7
            # numerical drift versus the bias-disabled baseline.
            if bool((corr_lambda_eff_candidate != 0).any().item()):
                from tools.correspondence_bias import compute_correspondence_bias
                corr_bias_base = compute_correspondence_bias(
                    states=states,
                    intrinsics=intrinsics,
                    grid_h=self.grid_height,
                    grid_w=self.grid_width,
                    cond_tokens=self.cond_tokens,
                    sigma_tokens=self.correspondence_bias_sigma_tokens,
                ).to(dtype=attn_mask.dtype if attn_mask is not None else x_seq.dtype,
                     device=x_seq.device)
                corr_lambda_eff = corr_lambda_eff_candidate
                corr_apply_idx_map = {
                    block_idx: pos
                    for pos, block_idx in enumerate(self.correspondence_bias_apply_layers)
                }

        def _per_block_mask(block_idx: int):
            """Return the additive attn_mask for block ``block_idx``: the
            base causal mask, plus the lambda-scaled correspondence bias when
            the layer is in ``correspondence_bias_apply_layers``.

            Shape contract: the returned mask is either ``None`` or shape
            ``(B, 1, N, N)`` so that ``F.scaled_dot_product_attention``
            broadcasts it across heads of the ``(B, num_heads, N, D)``
            q/k/v tensors. Returning a ``(B, N, N)`` tensor instead leads
            SDPA to align dim 1 (B) with the head dim and raise a shape
            mismatch.
            """
            if corr_bias_base is None or block_idx not in corr_apply_idx_map:
                return attn_mask
            lam = corr_lambda_eff[corr_apply_idx_map[block_idx]]
            extra = corr_bias_base * lam                  # (B, N, N)
            extra = extra.unsqueeze(1)                    # (B, 1, N, N)
            if attn_mask is None:
                return extra
            # attn_mask is (N, N); broadcast to (1, 1, N, N) so the sum
            # remains (B, 1, N, N).
            return attn_mask.unsqueeze(0).unsqueeze(0) + extra

        # -- transformer forward pass
        capture_rayqk_attention = bool(getattr(self, "_capture_rayqk_attention", False))
        requested_rayqk_attention_layers = set(
            int(v) for v in getattr(self, "_rayqk_attention_layers", ())
        )
        rayqk_attention_target_tokens = list(
            int(v) for v in getattr(self, "_rayqk_attention_target_tokens", ())
        )
        for block_idx, blk in enumerate(self.predictor_blocks):
            blk_attn_mask = _per_block_mask(block_idx)
            qk_mod = None
            if coord_qk_seq is not None and block_idx in coord_qk_apply_idx_map:
                gamma = self.coord_qk_gamma[coord_qk_apply_idx_map[block_idx]]
                qk_mod = coord_qk_seq * gamma.to(dtype=coord_qk_seq.dtype)
            if ray_qk_seq is not None and block_idx in ray_qk_apply_idx_map:
                gamma = self.ray_qk_gamma[ray_qk_apply_idx_map[block_idx]]
                ray_mod = ray_qk_seq * gamma.to(dtype=ray_qk_seq.dtype)
                if ray_debug is not None:
                    ray_mod_frames = ray_mod.view(
                        B,
                        T,
                        self.cond_tokens + self.grid_height * self.grid_width,
                        D,
                    )
                    ray_debug.setdefault("ray_qk_blocks", []).append({
                        "block": int(block_idx),
                        "gamma": float(gamma.detach().float().cpu().item()),
                        "target_mod": ray_mod_frames[:, -1, self.cond_tokens:, :].detach(),
                    })
                qk_mod = ray_mod if qk_mod is None else qk_mod + ray_mod
            attn_debug_enabled = (
                capture_ray_debug
                and capture_rayqk_attention
                and isinstance(getattr(blk, "attn", None), ACRoPEAttention)
                and int(block_idx) in requested_rayqk_attention_layers
            )
            if attn_debug_enabled:
                setattr(blk.attn, "_rayqk_debug_request", {
                    "block": int(block_idx),
                    "target_tokens": rayqk_attention_target_tokens,
                })
                setattr(blk.attn, "_last_rayqk_debug_payload", None)
            elif isinstance(getattr(blk, "attn", None), ACRoPEAttention):
                setattr(blk.attn, "_rayqk_debug_request", None)
                setattr(blk.attn, "_last_rayqk_debug_payload", None)
            if self.use_activation_checkpointing:
                if adaln_cond is not None:
                    qk_arg = qk_mod if qk_mod is not None else torch.zeros_like(x_seq)

                    def _adaln_checkpoint(x_in, cond_in, qk_in, mask_in):
                        return self._forward_adaln_block(
                            blk,
                            block_idx,
                            x_in,
                            cond_in,
                            mask=None,
                            attn_mask=mask_in,
                            T=T,
                            H=self.grid_height,
                            W=self.grid_width,
                            action_tokens=self.cond_tokens,
                            qk_mod=qk_in,
                        )

                    x_seq = torch.utils.checkpoint.checkpoint(
                        _adaln_checkpoint,
                        x_seq,
                        adaln_cond,
                        qk_arg,
                        blk_attn_mask.clone() if blk_attn_mask is not None else None,
                        use_reentrant=False,
                    )
                else:
                    x_seq = torch.utils.checkpoint.checkpoint(
                        blk,
                        x_seq,
                        None,           # mask
                        blk_attn_mask.clone() if blk_attn_mask is not None else None,
                        T,
                        self.grid_height,
                        self.grid_width,
                        self.cond_tokens,
                        qk_mod,
                        use_reentrant=False,
                    )
            else:
                if adaln_cond is not None:
                    x_seq = self._forward_adaln_block(
                        blk,
                        block_idx,
                        x_seq,
                        adaln_cond,
                        mask=None,
                        attn_mask=blk_attn_mask,
                        T=T,
                        H=self.grid_height,
                        W=self.grid_width,
                        action_tokens=self.cond_tokens,
                        qk_mod=qk_mod,
                    )
                else:
                    x_seq = blk(
                        x_seq,
                        mask=None,
                        attn_mask=blk_attn_mask,
                        T=T,
                        H=self.grid_height,
                        W=self.grid_width,
                        action_tokens=self.cond_tokens,
                        qk_mod=qk_mod,
                    )
            if attn_debug_enabled and ray_debug is not None:
                payload = getattr(blk.attn, "_last_rayqk_debug_payload", None)
                if payload is not None:
                    ray_debug.setdefault("ray_qk_attention_debug", []).append(payload)
                setattr(blk.attn, "_rayqk_debug_request", None)
                setattr(blk.attn, "_last_rayqk_debug_payload", None)
            if self.camera_ucpe_enabled and block_idx in self.camera_ucpe_apply_layers:
                if ray_embed is None:
                    raise RuntimeError("camera_ucpe_enabled=True requires ray_embed in forward().")
                x_seq = self.camera_ucpe_branches[str(block_idx)](
                    x_seq,
                    ray_embed.flatten(1, 2),
                    T,
                    self.grid_height * self.grid_width,
                    self.cond_tokens,
                )

        # -- split conditioning slots from visual patch slots
        #    layout per frame: [cond_0 .. cond_{K-1} | patch_0 .. patch_{HW-1}]
        x_seq = x_seq.view(B, T, self.cond_tokens + self.grid_height * self.grid_width, D)
        if ray_debug is not None:
            ray_debug["post_transformer_target_visual"] = x_seq[:, -1, self.cond_tokens:, :].detach()
        x_visual = x_seq[:, :, self.cond_tokens:, :].flatten(1, 2)   # [B, T*H*W, D]
        x_visual = self.predictor_norm(x_visual)
        if ray_debug is not None:
            ray_debug["post_norm_target_visual"] = x_visual.view(B, T, HW, D)[:, -1].detach()

        # Residual head (optional) refines `x_visual` before it feeds every
        # output projection. Sharing the refined features across the primary,
        # context, and delta heads keeps the three outputs consistent; earlier
        # revisions projected `x_visual` twice in the `use_residual_head=True`
        # path (once wasted) and fed the unrefined tensor to the context/delta
        # heads, producing a silent asymmetry between the heads.
        head_input = x_visual
        if self.use_residual_head:
            head_input = x_visual + self.residual_gate * self.residual_refine(x_visual)

        predictions = torch.cat([head(head_input) for head in self.predictor_proj], dim=-1)

        context_predictions = None
        if self.predict_all:
            context_predictions = torch.cat([head(head_input) for head in self.predictor_proj_context], dim=-1)

        delta_predictions = None
        if self.use_delta_head:
            delta_predictions = torch.cat([head(head_input) for head in self.predictor_proj_delta], dim=-1)

        fsq_logits = None
        if self.use_fsq_head:
            # Categorical FSQ head. Shares `head_input` with the continuous
            # heads so predictor_proj / predictor_proj_context / predictor_proj_delta
            # and fsq_head see identical features; callers train whichever
            # losses they want on top.
            fsq_logits = self.fsq_head(head_input)

        if ray_debug is not None:
            self._last_ray_conditioning = ray_debug

        return predictions, context_predictions, delta_predictions, fsq_logits, warped_context_raw


def vit_camera_ac_predictor(**kwargs):
    model = CameraConditionedPredictorAC(
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        **kwargs,
    )
    return model
