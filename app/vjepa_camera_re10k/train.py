# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.
#
# Camera-Pose Conditioned V-JEPA Training on RE10K
# ------------------------------------------------
# Adapts vjepa_droid/train.py to:
#   1. Load RE10K sequences via RE10KSequenceDataset (strided temporal clips)
#   2. Use CameraConditionedPredictorAC (camera_ac_predictor.py)
#   3. Call target_encoder with training_mode=True for multi-layer features
#   4. Compute V-JEPA 2.1 hierarchical dense predictive loss

import contextlib
import json
import logging
import os
import glob
import copy
import gc
import random
import shutil
import sys
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_camera_re10k.utils import (
    init_opt,
    init_video_model,
    load_checkpoint,
    load_pretrained,
)
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.training.visualization import visualize_pca_features, plot_training_curves

# FSQ categorical head wiring. Imports guarded so that if GWM's src/ is not on
# the path (standalone vjepa2 usage) this training script still parses — the
# FSQ branch is only reached when the YAML enables `model.use_fsq_head`.
try:
    from src.training.common.fsq import (  # type: ignore
        categorical_loss,
        load_hierarchical_fsq,
    )
    _FSQ_AVAILABLE = True
except Exception:
    categorical_loss = None  # type: ignore
    load_hierarchical_fsq = None  # type: ignore
    _FSQ_AVAILABLE = False

try:
    from src.training.common.depth_teacher import (  # type: ignore
        compute_teacher_depths_batch_with_confidence,
        load_depth_teacher,
    )
    _DEPTH_TEACHER_AVAILABLE = True
except Exception:
    compute_teacher_depths_batch_with_confidence = None  # type: ignore
    load_depth_teacher = None  # type: ignore
    _DEPTH_TEACHER_AVAILABLE = False

# Phase C — inference-time latent corrector. Local import; the module lives in
# the same vjepa2 subtree as the predictor, so this never falls back. See
# ``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` §5.
from src.models.latent_corrector import build_latent_corrector  # noqa: E402
from src.models.latent_patchgan import (  # noqa: E402
    LatentPatchDiscriminator,
    latent_patchgan_hinge_discriminator_loss,
    latent_patchgan_hinge_generator_loss,
)

log_timings = True
log_freq = 100
CHECKPOINT_FREQ = 10
VIZ_FREQ = 10
GARBAGE_COLLECT_ITR_FREQ = 50
MIN_FREE_BYTES_FOR_FULL_CHECKPOINT = 6 * 1024**3
CHECKPOINT_SAVE_BUFFER_BYTES = 4 * 1024**3
FULL_CHECKPOINT_SIZE_SAFETY_FACTOR = 1.10

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _quat_conjugate(quat):
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


def _motion_magnitude(
    action_step: torch.Tensor,
    mode: str = "translation",
) -> torch.Tensor:
    """Compute a per-sample scalar motion magnitude from a single-step action.

    Args:
        action_step: ``(B, 7)`` tensor laid out as ``[tx, ty, tz, qx, qy, qz, qw]``
            — the same action indexing convention used by ``_canonicalize_states``
            and ``_rotation_degrees_from_quat``.
        mode:
          * ``"translation"`` — L2 norm of the translation component.
          * ``"rotation"``    — rotation angle in radians recovered from the
            quaternion via ``2 * atan2(||xyz||, |w|)``, matching
            ``_rotation_degrees_from_quat``.
          * ``"total"``       — ``translation + rotation`` on roughly comparable
            scales. Callers can adjust sensitivity via ``motion_weight_alpha``.

    Returns:
        ``(B,)`` non-negative tensor. The scale is in scene units: meters for
        translation (RE10K poses are already metric-like) and radians for
        rotation.
    """
    if action_step.shape[-1] != 7:
        raise ValueError(
            f"_motion_magnitude expects 7-DOF actions (tx,ty,tz,qx,qy,qz,qw); "
            f"got last dim {action_step.shape[-1]}."
        )
    trans = torch.linalg.norm(action_step[..., :3], dim=-1)
    if mode == "translation":
        return trans
    imag_norm = torch.linalg.norm(action_step[..., 3:6], dim=-1)
    real = action_step[..., 6].abs().clamp_min(1.0e-8)
    rot = 2.0 * torch.atan2(imag_norm, real)
    if mode == "rotation":
        return rot
    if mode == "total":
        return trans + rot
    raise ValueError(
        f"motion_weight_mode must be 'translation', 'rotation', or 'total'; "
        f"got {mode!r}."
    )


def _motion_sample_weights(
    action_step: torch.Tensor,
    mode: str,
    alpha: float,
    eps: float = 1.0e-3,
) -> torch.Tensor:
    """Per-sample loss weights that amplify high-motion steps.

    Steps:
      1. Compute raw motion magnitude via ``_motion_magnitude``.
      2. Raise to ``alpha`` (alpha=0 disables the effect, alpha=1 is linear,
         alpha=2 quadratic emphasis).
      3. Normalize so ``mean(weights) == 1``. This keeps the overall L1 loss
         on the same scale as the unweighted baseline; only the relative
         distribution across batch samples changes.

    Returns ``(B,)`` with mean close to 1.
    """
    mag = _motion_magnitude(action_step, mode=mode)
    mag = (mag + eps) ** float(alpha)
    return mag / mag.mean().clamp_min(eps)


def _sigreg_match_to_target(
    preds: torch.Tensor,
    tgt: torch.Tensor,
    n_directions: int,
    n_layers: int,
) -> torch.Tensor:
    """SIGReg applied asymmetrically to predicted rollout latents.

    Implements the Sketched-Isotropic-Gaussian Regularizer from LeWorldModel
    (LeWM, arXiv:2603.19312 App. A) adapted to a frozen-encoder JEPA setup:
    ``preds`` are pre-whitened using the per-channel mean / std of the
    detached GT latent ``tgt`` so the SIGReg target distribution N(0, 1)
    is consistent with whatever distribution V-JEPA 2.1 actually outputs
    on re10k frames. The loss penalizes deviation of the whitened preds
    from N(0, 1) via the univariate Epps-Pulley test statistic on M random
    unit-direction 1D projections (Cramer-Wold reduction).

    The key difference from FSQ's CE-on-codes auxiliary is *what* gets
    regularized: FSQ shapes the predictor output toward discrete code
    centroids (output-side), while SIGReg here shapes the *self-fed input
    distribution* the next rollout step sees toward the real V-JEPA
    latent distribution. This directly targets the closed-loop
    input-distribution mismatch identified as the root cause of the
    H20 rollout gap.

    Args:
        preds: ``(B, HW, n_layers * D_layer)`` predicted latent at a
            rollout step. Gradients flow into the predictor.
        tgt: ``(B, HW, n_layers * D_layer)`` GT target latent at the same
            step; detached before statistics are taken (the frozen V-JEPA
            encoder is not re-shaped).
        n_directions: ``M`` — number of random unit projections per layer.
            ~64 is enough in practice (LeWM uses similar magnitudes).
        n_layers: Hierarchical layer count; ``preds`` / ``tgt`` are split
            along the channel axis and SIGReg is applied per-layer, then
            averaged. Matches the per-layer structure of the main L1 loss.

    Returns:
        Scalar loss (mean over layers).
    """
    D_total = preds.shape[-1]
    assert D_total % n_layers == 0, (
        f"preds.shape[-1]={D_total} not divisible by n_layers={n_layers}"
    )
    D_layer = D_total // n_layers
    pred_chunks = preds.split(D_layer, dim=-1)
    tgt_chunks = tgt.detach().split(D_layer, dim=-1)

    # Epps-Pulley trapezoid quadrature on t in [0.2, 4.0], 16 nodes
    # (LeWM App. A recommended range). φ_0(t) = exp(-t^2/2) is the
    # characteristic function of N(0, 1); w(t) with lambda=1 is the
    # standard weighting used in LeWM.
    t_nodes = torch.linspace(
        0.2, 4.0, 16, device=preds.device, dtype=preds.dtype
    )
    dt = (4.0 - 0.2) / 15.0
    phi_0 = torch.exp(-0.5 * t_nodes ** 2)
    w = torch.exp(-0.5 * t_nodes ** 2)

    total = preds.new_zeros(())
    for p, t_ in zip(pred_chunks, tgt_chunks):
        with torch.no_grad():
            mu = t_.mean(dim=(0, 1), keepdim=True)
            sigma = t_.std(dim=(0, 1), keepdim=True).clamp_min(1.0e-6)
        p_white = (p - mu) / sigma
        B_, N_, D = p_white.shape
        # Unit-normed random directions on S^{D-1}.
        u = torch.randn(D, n_directions, device=p.device, dtype=p.dtype)
        u = u / u.norm(dim=0, keepdim=True).clamp_min(1.0e-6)
        h = p_white.reshape(-1, D) @ u  # (B*N, M)
        # Empirical characteristic function per projection along t nodes.
        h_t = h.unsqueeze(-1) * t_nodes  # (B*N, M, 16)
        phi_N_real = torch.cos(h_t).mean(dim=0)  # (M, 16)
        phi_N_imag = torch.sin(h_t).mean(dim=0)
        diff_sq = (phi_N_real - phi_0).pow(2) + phi_N_imag.pow(2)
        ep = (diff_sq * w).sum(dim=-1) * dt  # (M,)
        total = total + ep.mean()
    return total / float(n_layers)


def _quat_multiply(quat_a, quat_b):
    ax, ay, az, aw = quat_a.unbind(dim=-1)
    bx, by, bz, bw = quat_b.unbind(dim=-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


def _quat_rotate(quat, vec):
    quat_xyz = quat[..., :3]
    quat_w = quat[..., 3:4]
    cross_term = 2.0 * torch.cross(quat_xyz, vec, dim=-1)
    return vec + quat_w * cross_term + torch.cross(quat_xyz, cross_term, dim=-1)


def _invert_action_se3(action: torch.Tensor) -> torch.Tensor:
    """Invert a 7D SE(3) action ``[tx, ty, tz, qx, qy, qz, qw]``.

    If ``action`` encodes a transformation ``T = (R(q), t)`` acting as
    ``T(p) = R(q) p + t``, then ``T^{-1} = (R(q)^T, -R(q)^T t)`` with
    ``q_inv = q_conjugate`` (XYZW convention).

    Supports arbitrary leading batch dims. Used by the reversibility cycle
    loss (Phase R) and the depth-gated projective transport (Phase T); see
    ``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` section 3.1.
    """
    t = action[..., :3]
    q = F.normalize(action[..., 3:], dim=-1, eps=1.0e-6)
    q_inv = _quat_conjugate(q)
    t_inv = -_quat_rotate(q_inv, t)
    # Keep w >= 0 sign convention, matching _canonicalize_states.
    q_inv = torch.where(q_inv[..., 3:4] < 0, -q_inv, q_inv)
    return torch.cat([t_inv, q_inv], dim=-1)


def _rotation_unseen_boundary_mask(
    predictor_ref,
    local_states_canon: torch.Tensor,
    local_intrinsics_window: torch.Tensor | None,
    hw: int,
    dtype: torch.dtype,
    band: float,
) -> torch.Tensor | None:
    """Soft mask for target tokens that become unseen near the source border."""
    if local_intrinsics_window is None or local_states_canon.shape[1] < 2:
        return None
    gh = int(getattr(predictor_ref, "grid_height", 0))
    gw = int(getattr(predictor_ref, "grid_width", 0))
    if gh <= 0 or gw <= 0 or gh * gw != hw:
        return None

    src_state = local_states_canon[:, -2].to(dtype=dtype)
    tgt_state = local_states_canon[:, -1].to(dtype=dtype)
    src_intr = local_intrinsics_window[:, -2].to(dtype=dtype)
    tgt_intr = local_intrinsics_window[:, -1].to(dtype=dtype)

    R_src = predictor_ref._quat_xyzw_to_rotmat(src_state[..., 3:7])
    R_tgt = predictor_ref._quat_xyzw_to_rotmat(tgt_state[..., 3:7])
    K_src = predictor_ref._build_K_from_normalized(src_intr)
    K_tgt_inv = predictor_ref._build_K_inv_from_normalized(tgt_intr)

    y_norm = (torch.arange(gh, device=src_state.device, dtype=dtype) + 0.5) / gh
    x_norm = (torch.arange(gw, device=src_state.device, dtype=dtype) + 0.5) / gw
    yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
    pixel_grid_tgt = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.ones(hw, device=src_state.device, dtype=dtype)],
        dim=-1,
    )

    R_st = torch.bmm(R_src.transpose(-1, -2), R_tgt)
    H_inv = torch.bmm(K_src, torch.bmm(R_st, K_tgt_inv))
    src_h = torch.einsum("bij,nj->bni", H_inv, pixel_grid_tgt)

    eps = 1.0e-6
    src_xy = src_h[..., :2] / src_h[..., 2:3].clamp_min(eps)
    uv_x = src_xy[..., 0:1]
    uv_y = src_xy[..., 1:2]
    valid = (
        (src_h[..., 2:3] > eps)
        & (uv_x >= 0.0) & (uv_x <= 1.0)
        & (uv_y >= 0.0) & (uv_y <= 1.0)
    ).to(dtype)
    border = torch.minimum(
        torch.minimum(uv_x, 1.0 - uv_x),
        torch.minimum(uv_y, 1.0 - uv_y),
    ).clamp(-0.5, 0.5)
    band = max(float(band), 1.0e-6)
    outside_near_boundary = ((border + band) / band).clamp(0.0, 1.0)
    return (1.0 - valid) * outside_near_boundary


def _canvas_rot_weight(
    local_actions_window: torch.Tensor,
    dtype: torch.dtype,
    eps: float = 1.0e-6,
) -> torch.Tensor:
    """Compute per-sample rotation dominance score from the incoming action.

    Returns a (B,) tensor in [0, 1] where 1.0 = pure rotation, 0.0 = pure
    translation.  Used by canvas_warp_mode='rot_gated' to scale beta_valid.

    The incoming action into the target step is local_actions_window[:, -2]
    (dataset convention: actions[t] encodes t→t+1; the last slot is zero-
    padded, so -2 is the last real action).

    Rotation magnitude: geodesic angle = 2 * arccos(|qw|).
    Translation magnitude: Euclidean norm of [tx, ty, tz].
    """
    if local_actions_window is None or local_actions_window.shape[1] < 2:
        B = local_actions_window.shape[0] if local_actions_window is not None else 1
        return torch.ones(B, dtype=dtype)
    action = local_actions_window[:, -2].to(dtype=dtype)  # (B, 7)
    t_mag = action[:, :3].norm(dim=-1)                    # (B,)
    qw = action[:, 6].clamp(-1.0 + eps, 1.0 - eps)
    rot_angle = 2.0 * torch.arccos(qw.abs())              # (B,) in [0, pi]
    rot_score = rot_angle / (rot_angle + t_mag + eps)
    return rot_score.clamp(0.0, 1.0)


def _depth_visibility_region_masks(
    predictor_ref,
    local_states_canon: torch.Tensor,
    local_intrinsics_window: torch.Tensor | None,
    source_depth: torch.Tensor,
    target_depth: torch.Tensor,
    hw: int,
    dtype: torch.dtype,
    depth_tau: float,
    boundary_dilate: int,
) -> dict[str, torch.Tensor] | None:
    """Depth-aware visible/boundary/disoccluded masks on target token centers."""
    if local_intrinsics_window is None or local_states_canon.shape[1] < 2:
        return None
    gh = int(getattr(predictor_ref, "grid_height", 0))
    gw = int(getattr(predictor_ref, "grid_width", 0))
    if gh <= 0 or gw <= 0 or gh * gw != hw:
        return None

    device = local_states_canon.device
    src_state = local_states_canon[:, -2].to(dtype=dtype)
    tgt_state = local_states_canon[:, -1].to(dtype=dtype)
    src_intr = local_intrinsics_window[:, -2].to(dtype=dtype)
    tgt_intr = local_intrinsics_window[:, -1].to(dtype=dtype)

    def _to_bhw(depth: torch.Tensor) -> torch.Tensor:
        depth = depth.to(device=device, dtype=dtype)
        if depth.ndim == 4 and depth.shape[1] == 1:
            depth = depth[:, 0]
        if depth.ndim != 3:
            raise ValueError(f"Expected depth shape (B,H,W), got {tuple(depth.shape)}")
        return depth

    source_depth = _to_bhw(source_depth)
    target_depth = _to_bhw(target_depth)
    B = int(target_depth.shape[0])
    target_tok = F.interpolate(
        target_depth.unsqueeze(1).float(),
        size=(gh, gw),
        mode="bilinear",
        align_corners=False,
    ).to(dtype=dtype).reshape(B, hw)

    R_src = predictor_ref._quat_xyzw_to_rotmat(src_state[..., 3:7])
    R_tgt = predictor_ref._quat_xyzw_to_rotmat(tgt_state[..., 3:7])
    K_src = predictor_ref._build_K_from_normalized(src_intr)
    K_tgt_inv = predictor_ref._build_K_inv_from_normalized(tgt_intr)

    y_norm = (torch.arange(gh, device=device, dtype=dtype) + 0.5) / gh
    x_norm = (torch.arange(gw, device=device, dtype=dtype) + 0.5) / gw
    yy, xx = torch.meshgrid(y_norm, x_norm, indexing="ij")
    pixel_grid_tgt = torch.stack(
        [xx.reshape(-1), yy.reshape(-1), torch.ones(hw, device=device, dtype=dtype)],
        dim=-1,
    )

    rays_tgt = torch.einsum("bij,nj->bni", K_tgt_inv, pixel_grid_tgt)
    pts_cam_tgt = rays_tgt * target_tok.clamp_min(1.0e-3).unsqueeze(-1)
    pts_world = torch.einsum("bij,bnj->bni", R_tgt, pts_cam_tgt) + tgt_state[:, None, :3]
    pts_cam_src = torch.einsum(
        "bji,bnj->bni",
        R_src,
        pts_world - src_state[:, None, :3],
    )
    src_z = pts_cam_src[..., 2].clamp_min(1.0e-6)
    src_h = torch.einsum("bij,bnj->bni", K_src, pts_cam_src)
    src_xy = src_h[..., :2] / src_h[..., 2:3].clamp_min(1.0e-6)
    src_x = src_xy[..., 0]
    src_y = src_xy[..., 1]
    in_bounds = (
        (src_h[..., 2] > 1.0e-6)
        & (src_x >= 0.0) & (src_x <= 1.0)
        & (src_y >= 0.0) & (src_y <= 1.0)
    )

    sample_grid = (src_xy * 2.0 - 1.0).reshape(B, gh, gw, 2)
    sampled_source = F.grid_sample(
        source_depth.unsqueeze(1).float(),
        sample_grid.float(),
        mode="bilinear",
        padding_mode="zeros",
        align_corners=False,
    ).to(dtype=dtype).reshape(B, hw)
    finite = torch.isfinite(sampled_source) & torch.isfinite(target_tok) & (sampled_source > 1.0e-6)
    depth_err = torch.abs(torch.log(src_z) - torch.log(sampled_source.clamp_min(1.0e-6)))
    visible = (in_bounds & finite & (depth_err < float(depth_tau))).to(dtype)
    disoccluded = 1.0 - visible

    bd = max(1, int(boundary_dilate))
    kernel = 2 * bd + 1
    disocc_2d = disoccluded.reshape(B, 1, gh, gw)
    vis_2d = visible.reshape(B, 1, gh, gw)
    dil_disocc = F.max_pool2d(disocc_2d, kernel_size=kernel, stride=1, padding=bd)
    dil_vis = F.max_pool2d(vis_2d, kernel_size=kernel, stride=1, padding=bd)
    boundary = (dil_disocc * dil_vis).reshape(B, hw, 1).clamp(0.0, 1.0)
    disoccluded = (disoccluded.reshape(B, hw, 1) * (1.0 - boundary)).clamp(0.0, 1.0)
    visible = visible.reshape(B, hw, 1)
    return {
        "visible": visible,
        "boundary": boundary,
        "disoccluded": disoccluded,
    }


def _compose_action_se3(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Compose two 7D SE(3) actions.

    Returns ``c`` such that ``T_c(p) = T_a(T_b(p))`` (apply ``b`` first,
    then ``a``). Used for state re-canonicalization under reversed
    trajectories and for unit-testing the inverse helper.
    """
    t_a, q_a = a[..., :3], F.normalize(a[..., 3:], dim=-1, eps=1.0e-6)
    t_b, q_b = b[..., :3], F.normalize(b[..., 3:], dim=-1, eps=1.0e-6)
    t_c = _quat_rotate(q_a, t_b) + t_a
    q_c = _quat_multiply(q_a, q_b)
    q_c = F.normalize(q_c, dim=-1, eps=1.0e-6)
    q_c = torch.where(q_c[..., 3:4] < 0, -q_c, q_c)
    return torch.cat([t_c, q_c], dim=-1)


def _canonicalize_states(states):
    ref_trans = states[:, :1, :3]
    ref_quat = F.normalize(states[:, :1, 3:], dim=-1, eps=1.0e-6)
    inv_ref_quat = _quat_conjugate(ref_quat)
    inv_ref_quat = inv_ref_quat.expand(-1, states.size(1), -1)
    rel_trans = _quat_rotate(inv_ref_quat, states[:, :, :3] - ref_trans)
    rel_quat = _quat_multiply(inv_ref_quat, F.normalize(states[:, :, 3:], dim=-1, eps=1.0e-6))
    rel_quat = F.normalize(rel_quat, dim=-1, eps=1.0e-6)
    rel_quat = torch.where(rel_quat[..., 3:4] < 0, -rel_quat, rel_quat)
    return torch.cat([rel_trans, rel_quat], dim=-1)


def _init_re10k_loader(
    data_root,
    seq_len,
    stride,
    min_stride,
    stride_values,
    image_size,
    batch_size,
    num_workers,
    pin_mem,
    persistent_workers,
    world_size,
    rank,
    manifest_cache_dir=None,
    fixed_manifest_path=None,
    local_chunk_cache_dir=None,
    local_chunk_cache_limit_gb=0.0,
):
    """Build RE10K train DataLoader using RE10KLazySceneDataset + RE10KSequenceDataset."""
    # Import here so the module can be used without GWM on sys.path when running
    # from inside the submodule (append parent if needed).
    gwm_root = os.path.join(os.path.dirname(__file__), "..", "..", "..", "..")
    gwm_root = os.path.abspath(gwm_root)
    if gwm_root not in sys.path:
        sys.path.insert(0, gwm_root)

    from src.core.dataset import RE10KLazySceneDataset
    from src.training.multiscene.re10k_sequence_dataset import (
        RE10KSequenceDataset,
        collate_re10k_sequences,
    )

    lazy_ds = RE10KLazySceneDataset(
        root=data_root,
        stage="train",
        image_size=image_size,
        manifest_cache_dir=manifest_cache_dir,
        fixed_manifest_path=fixed_manifest_path,
        local_chunk_cache_dir=local_chunk_cache_dir,
        local_chunk_cache_limit_gb=local_chunk_cache_limit_gb,
    )
    dataset = RE10KSequenceDataset(
        lazy_dataset=lazy_ds,
        seq_len=seq_len,
        stride=stride,
        min_stride=min_stride,
        stride_values=stride_values,
        image_size=image_size,
    )
    sampler = DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=True,
        drop_last=True,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=num_workers,
        pin_memory=pin_mem,
        persistent_workers=persistent_workers and num_workers > 0,
        collate_fn=collate_re10k_sequences,
        drop_last=True,
    )
    return loader, sampler


def _normalize_and_concat(h, embed_dim):
    """Layer-norm each of the 4 hierarchical chunks of a (B, N, 4*D) tensor."""
    n_layers = h.shape[-1] // embed_dim
    chunks = [
        F.layer_norm(h[..., i * embed_dim:(i + 1) * embed_dim], (embed_dim,))
        for i in range(n_layers)
    ]
    return torch.cat(chunks, dim=-1)


def _set_module_requires_grad(module, requires_grad: bool) -> None:
    if module is None:
        return
    target = module.module if hasattr(module, "module") else module
    for param in target.parameters():
        param.requires_grad_(requires_grad)


def main(args, resume_preempt=False):
    # ----------------------------------------------------------------------- #
    #  PASSED IN PARAMS FROM CONFIG FILE
    # ----------------------------------------------------------------------- #

    # -- META
    folder = args.get("folder")
    scratch_folder = args.get("scratch_folder") or folder
    cfgs_meta = args.get("meta")
    r_file = cfgs_meta.get("resume_checkpoint", None)
    p_file = cfgs_meta.get("pretrain_checkpoint", None)
    load_predictor = cfgs_meta.get("load_predictor", False)
    context_encoder_key = cfgs_meta.get("context_encoder_key", "encoder")
    target_encoder_key = cfgs_meta.get("target_encoder_key", "target_encoder")
    load_encoder = cfgs_meta.get("load_encoder", True)
    seed = cfgs_meta.get("seed", _GLOBAL_SEED)
    save_every_freq = cfgs_meta.get("save_every_freq", -1)
    checkpoint_save_mode = str(
        cfgs_meta.get("checkpoint_save_mode", os.environ.get("CHECKPOINT_SAVE_MODE", "auto"))
    ).lower()
    checkpoint_saving_enabled = checkpoint_save_mode not in {"none", "off", "disabled", "false", "0"}
    save_final_checkpoint = bool(cfgs_meta.get("save_final_checkpoint", True))
    checkpoint_eval_enabled = bool(cfgs_meta.get("checkpoint_eval_enabled", True))
    skip_batches = cfgs_meta.get("skip_batches", -1)
    use_sdpa = cfgs_meta.get("use_sdpa", True)
    sync_gc = cfgs_meta.get("sync_gc", False)
    which_dtype = cfgs_meta.get("dtype", "bfloat16")
    logger.info(f"{which_dtype=}")
    if which_dtype.lower() == "bfloat16":
        dtype = torch.bfloat16
        mixed_precision = True
    elif which_dtype.lower() == "float16":
        dtype = torch.float16
        mixed_precision = True
    else:
        dtype = torch.float32
        mixed_precision = False

    # -- MODEL
    cfgs_model = args.get("model")
    compile_model = cfgs_model.get("compile_model", False)
    use_activation_checkpointing = cfgs_model.get("use_activation_checkpointing", False)
    model_name = cfgs_model.get("model_name")
    # Encoder backbone (``vjepa21`` default; ``mast3r`` swaps to a frozen
    # MASt3REncoderAdapter — see thirdparty/vjepa2/app/vjepa_camera_re10k/
    # mast3r_encoder.py and the ``..._cycle_mast3r`` RUN_ONLY case in
    # scripts/train_camera_re10k.sh). The shell driver may override via
    # ENCODER_BACKBONE / ENCODER_FREEZE env vars (init_video_model honours
    # both).
    encoder_backbone = str(cfgs_model.get("encoder_backbone", "vjepa21")).lower().strip()
    encoder_freeze = bool(cfgs_model.get("encoder_freeze", False))
    pred_depth = cfgs_model.get("pred_depth")
    pred_num_heads = cfgs_model.get("pred_num_heads", None)
    pred_embed_dim = cfgs_model.get("pred_embed_dim")
    pred_is_frame_causal = cfgs_model.get("pred_is_frame_causal", True)
    uniform_power = cfgs_model.get("uniform_power", True)
    use_rope = cfgs_model.get("use_rope", True)
    use_silu = cfgs_model.get("use_silu", False)
    use_pred_silu = cfgs_model.get("use_pred_silu", False)
    wide_silu = cfgs_model.get("wide_silu", True)
    n_hierarchical_layers = cfgs_model.get("n_hierarchical_layers", 4)
    use_intrinsics = cfgs_model.get("use_intrinsics", True)
    predict_all = cfgs_model.get("predict_all", True)
    state_dim = cfgs_model.get("state_dim", 7)
    action_dim = cfgs_model.get("action_dim", 7)
    intrinsics_dim = cfgs_model.get("intrinsics_dim", 4)
    use_ray_pe = bool(cfgs_model.get("use_ray_pe", False))
    ray_pe_dim = int(cfgs_model.get("ray_pe_dim", 6))
    ray_pe_hidden = int(cfgs_model.get("ray_pe_hidden", 256))
    # RayMap v2 representation: ``origin_dir`` (legacy bit-identical), ``plucker``
    # (canonical 6D Plücker line representation), ``plucker_delta`` (12D with
    # per-frame delta vs anchor). See @docs/RAYMAP_V2_DESIGN.md.
    ray_pe_mode = str(cfgs_model.get("ray_pe_mode", "origin_dir"))
    # DA3 RayMap v3 (2026-04-30): when set to ``raymap_only`` the per-frame
    # state token is dropped; pose lives entirely in the dense per-patch
    # raymap PE. Requires ``use_ray_pe=True``. See @docs/RAYMAP_V3_DESIGN.md.
    pose_conditioning_mode = str(cfgs_model.get("pose_conditioning_mode", "token+raymap"))
    action_token_mode = str(cfgs_model.get("action_token_mode", "transition"))
    # RayMap v4c (2026-05-01): ray_visibility_features adds rotation-only warp
    # correspondence block (uv_warp, valid, border) to plucker_pair. See
    # @docs/CAMERA_PREDICTOR_FULL_HISTORY.md §4.6.
    ray_visibility_features = str(cfgs_model.get("ray_visibility_features", "none"))
    use_delta_head = bool(cfgs_model.get("use_delta_head", False))
    delta_loss_weight = float(cfgs_model.get("delta_loss_weight", 0.25))
    use_residual_head = bool(cfgs_model.get("use_residual_head", False))
    residual_head_depth = int(cfgs_model.get("residual_head_depth", 2))
    residual_head_ratio = float(cfgs_model.get("residual_head_ratio", 2.0))
    use_completion_head = bool(cfgs_model.get("use_completion_head", False))
    completion_head_depth = int(cfgs_model.get("completion_head_depth", 2))
    completion_head_ratio = float(cfgs_model.get("completion_head_ratio", 2.0))
    use_four_layer_birth_head = bool(cfgs_model.get("use_four_layer_birth_head", True))
    completion_head_hidden = int(cfgs_model.get("completion_head_hidden", 512))
    # Rotation-only homography warp of context-frame tokens into target grid
    # (see CameraConditionedPredictorAC._warp_context_tokens). Diagnostic
    # ablation: default off, gated by cfgs_model and/or env var.
    warp_context_latents = bool(cfgs_model.get("warp_context_latents", False))
    warp_context_padding_mode = str(cfgs_model.get("warp_context_padding_mode", "zeros"))
    # Path B-lite Step 1 — depth-aware projective warp.
    # ``warp_mode`` selects the transport: "auto" maps to legacy behaviour
    # (rotation_only iff warp_context_latents=True else off); "rotation_only"
    # forces the legacy infinity-plane homography; "projective_probe" uses
    # the depth-probe target depth and requires ``depth_probe_checkpoint``;
    # "projective_da3" uses online DA3 target depth supplied by the trainer.
    warp_mode = str(cfgs_model.get("warp_mode", "auto"))
    depth_probe_checkpoint = cfgs_model.get("depth_probe_checkpoint", None)
    if depth_probe_checkpoint is not None:
        depth_probe_checkpoint = str(depth_probe_checkpoint)
    target_slot_mode = str(cfgs_model.get("target_slot_mode", "mask")).lower().strip()
    canvas_beta_init_valid = float(cfgs_model.get("canvas_beta_init_valid", 0.0))
    canvas_beta_init_boundary = float(cfgs_model.get("canvas_beta_init_boundary", 0.0))
    canvas_beta_init_oov = float(cfgs_model.get("canvas_beta_init_oov", 0.0))
    canvas_use_mask_feat_embed = bool(cfgs_model.get("canvas_use_mask_feat_embed", True))
    canvas_use_token_type_embed = bool(cfgs_model.get("canvas_use_token_type_embed", True))
    canvas_use_target_step_embed = bool(cfgs_model.get("canvas_use_target_step_embed", True))
    canvas_warp_mode = str(cfgs_model.get("canvas_warp_mode", "raw"))
    canvas_projector_gamma_init = float(cfgs_model.get("canvas_projector_gamma_init", 0.0))
    canvas_norm_clip_ratio = float(cfgs_model.get("canvas_norm_clip_ratio", 1.0))
    canvas_block_mask_enabled = bool(cfgs_model.get("canvas_block_mask_enabled", False))
    canvas_block_mask_min_rects = int(cfgs_model.get("canvas_block_mask_min_rects", 1))
    canvas_block_mask_max_rects = int(cfgs_model.get("canvas_block_mask_max_rects", 3))
    canvas_block_mask_min_frac = float(cfgs_model.get("canvas_block_mask_min_frac", 0.2))
    canvas_block_mask_max_frac = float(cfgs_model.get("canvas_block_mask_max_frac", 0.3))
    # LM7: use DA3 depth warp boundary as canvas boundary mask instead of the
    # rotation-homography approximation.  Requires DA3 depth to be loaded
    # (triggered automatically when canvas_boundary_source == "da3").
    canvas_boundary_source = str(cfgs_model.get("canvas_boundary_source", "rotation"))
    camera_ucpe_enabled = bool(cfgs_model.get("camera_ucpe_enabled", False))
    camera_ucpe_apply_layers = tuple(int(i) for i in cfgs_model.get("camera_ucpe_apply_layers", [0, 3, 6, 9]))
    camera_ucpe_gamma_init = float(cfgs_model.get("camera_ucpe_gamma_init", 0.0))
    predictor_adaln_enabled = bool(cfgs_model.get("predictor_adaln_enabled", False))
    predictor_adaln_hidden = int(cfgs_model.get("predictor_adaln_hidden", 512))
    use_posthoc_completion_blend = bool(cfgs_model.get("use_posthoc_completion_blend", True))

    # ------------------------------------------------------------------ #
    # Phase E1 — rotation-homography correspondence attention bias.
    # See ``@/home/ak/GWM/docs/CORRESPONDENCE_BIAS_E1.md``. All defaults
    # are off so pre-correspondence runs are bit-identical.
    # ------------------------------------------------------------------ #
    cfgs_corr_bias = cfgs_model.get("correspondence_bias", {}) or {}
    correspondence_bias_enabled = bool(cfgs_corr_bias.get("enabled", False))
    correspondence_bias_mode = str(cfgs_corr_bias.get("mode", "rotation_homography"))
    correspondence_bias_sigma_tokens = float(cfgs_corr_bias.get("sigma_tokens", 2.0))
    correspondence_bias_lambda_init = float(cfgs_corr_bias.get("lambda_init", 0.0))
    correspondence_bias_learnable = bool(cfgs_corr_bias.get("learnable", True))
    correspondence_bias_apply_layers = tuple(int(i) for i in cfgs_corr_bias.get(
        "apply_layers", (0, 1, 2, 3, 4, 5),
    ))
    correspondence_bias_start_epoch = int(cfgs_corr_bias.get("start_epoch", 0))
    correspondence_bias_ramp_end_epoch = int(cfgs_corr_bias.get("ramp_end_epoch", 5))
    if correspondence_bias_enabled:
        logger.info(
            "Phase E1 correspondence bias: enabled mode=%s sigma_tokens=%.2f "
            "lambda_init=%.4f learnable=%s apply_layers=%s "
            "start_epoch=%d ramp_end_epoch=%d",
            correspondence_bias_mode, correspondence_bias_sigma_tokens,
            correspondence_bias_lambda_init, correspondence_bias_learnable,
            correspondence_bias_apply_layers,
            correspondence_bias_start_epoch, correspondence_bias_ramp_end_epoch,
        )
    else:
        logger.info("Phase E1 correspondence bias: disabled.")
    # -- FSQ categorical head (optional). Enabled by `model.use_fsq_head: true`.
    # Pretrained FSQ tokenizer path is read from meta so the FSQ weights can
    # live alongside the pretrain checkpoint without polluting model config.
    use_fsq_head = bool(cfgs_model.get("use_fsq_head", False))
    fsq_loss_weight = float(cfgs_model.get("fsq_loss_weight", 0.5))
    fsq_ckpt_path = cfgs_meta.get("fsq_ckpt_path", None)
    fsq_module = None
    fsq_total_code_axes = 0
    fsq_levels = 0
    if use_fsq_head:
        if not _FSQ_AVAILABLE:
            raise ImportError(
                "model.use_fsq_head=True requires src.training.common.fsq to be "
                "importable. Check that GWM repo root is on sys.path."
            )
        if fsq_ckpt_path is None:
            raise ValueError(
                "model.use_fsq_head=True but meta.fsq_ckpt_path is not set. "
                "Point it at a checkpoint produced by tools/pretrain_fsq.py."
            )
        fsq_module = load_hierarchical_fsq(fsq_ckpt_path, map_location="cpu")
        fsq_total_code_axes = fsq_module.total_code_axes
        fsq_levels = fsq_module.levels
        logger.info(
            f"Loaded HierarchicalFSQ from {fsq_ckpt_path}: "
            f"n_layers={fsq_module.n_layers} per_layer_dim={fsq_module.per_layer_dim} "
            f"bottleneck_dim={fsq_module.bottleneck_dim} levels={fsq_levels} "
            f"total_code_axes={fsq_total_code_axes} fsq_loss_weight={fsq_loss_weight}"
        )

    # -- DATA
    cfgs_data = args.get("data")
    data_root = cfgs_data.get("data_root")
    seq_len = cfgs_data.get("seq_len", 8)
    stride = cfgs_data.get("stride", 4)
    min_stride = cfgs_data.get("min_stride", 1)
    stride_values = cfgs_data.get("stride_values", None)
    if stride_values in (None, ""):
        stride_values = None
    else:
        if not isinstance(stride_values, (list, tuple)):
            raise ValueError("data.stride_values must be a list or tuple of positive integers.")
        stride_values = [max(1, int(v)) for v in stride_values]
        if len(stride_values) == 0:
            stride_values = None
    stride_candidates = sorted(set(stride_values)) if stride_values is not None else list(range(max(1, int(min_stride)), max(int(stride), int(min_stride)) + 1))
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size", 1)
    total_tubelets = seq_len // tubelet_size
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    pin_mem = cfgs_data.get("pin_mem", True)
    num_workers = cfgs_data.get("num_workers", 8)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    # Number of context tubelets shown to the context encoder.
    # The predictor must hallucinate the remaining (T_tok - n_ctx_tubelets) tubelets.
    n_ctx_tubelets = cfgs_data.get("n_ctx_tubelets", 1)
    ar_random_context = bool(cfgs_data.get("ar_random_context", False))
    ar_random_local_window = bool(cfgs_data.get("ar_random_local_window", False))
    min_ctx_tubelets = int(cfgs_data.get("min_ctx_tubelets", 1))
    max_ctx_tubelets = cfgs_data.get("max_ctx_tubelets", None)
    if max_ctx_tubelets in (None, ""):
        max_ctx_tubelets = total_tubelets - 1
    else:
        max_ctx_tubelets = int(max_ctx_tubelets)
    if total_tubelets < 2:
        raise ValueError(
            f"Camera autoregressive training requires at least 2 tubelets, got seq_len={seq_len}, tubelet_size={tubelet_size}."
        )
    max_valid_ctx_tubelets = total_tubelets - 1
    n_ctx_tubelets = max(1, min(int(n_ctx_tubelets), max_valid_ctx_tubelets))
    min_ctx_tubelets = max(1, min(min_ctx_tubelets, max_valid_ctx_tubelets))
    max_ctx_tubelets = max(1, min(max_ctx_tubelets, max_valid_ctx_tubelets))
    if min_ctx_tubelets > max_ctx_tubelets:
        raise ValueError(
            f"Invalid autoregressive context range: min_ctx_tubelets={min_ctx_tubelets} > max_ctx_tubelets={max_ctx_tubelets}."
        )
    # Manifest / cache paths (forwarded to RE10KLazySceneDataset)
    manifest_cache_dir = cfgs_data.get("manifest_cache_dir", None)
    fixed_manifest_path = cfgs_data.get("fixed_manifest_path", None)
    local_chunk_cache_dir = cfgs_data.get("local_chunk_cache_dir", None)
    local_chunk_cache_limit_gb = float(cfgs_data.get("local_chunk_cache_limit_gb", 0.0))
    eval_fixed_manifest_path = cfgs_data.get("eval_fixed_manifest_path", None)
    n_pca_scenes = int(cfgs_data.get("n_pca_scenes", 4))
    n_motion_eval_scenes = int(cfgs_data.get("n_motion_eval_scenes", max(64, n_pca_scenes)))
    n_fixed_stride_eval_scenes = int(cfgs_data.get("n_fixed_stride_eval_scenes", max(32, min(128, n_motion_eval_scenes))))
    eval_stride_small = int(cfgs_data.get("eval_stride_small", stride_candidates[0]))
    eval_stride_medium = int(cfgs_data.get("eval_stride_medium", stride_candidates[len(stride_candidates) // 2]))
    eval_stride_large = int(cfgs_data.get("eval_stride_large", stride_candidates[-1]))

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp", 1.0)
    normalize_reps = cfgs_loss.get("normalize_reps", True)
    # Per-layer balance rescales each hierarchical chunk's L1 contribution by
    # the target chunk's std. This compensates for the gradient imbalance
    # across hierarchical layers that appears once LayerNorm is removed from
    # the loss (normalize_reps=False): without rescaling, high-magnitude deep
    # layers dominate and early layers are effectively untrained. Default
    # tracks normalize_reps — on when LN is off, off when LN is on.
    per_layer_balance = bool(
        cfgs_loss.get("per_layer_balance", not normalize_reps)
    )
    cfgs_intermediate = cfgs_loss.get("intermediate_supervision", {}) or {}
    intermediate_supervision_enabled = bool(
        cfgs_intermediate.get("enabled", False)
    )
    _default_out_layers = [5, 11, 17, 23][:n_hierarchical_layers]
    _out_layers_env = os.environ.get("VJEPA_OUT_LAYERS", "")
    if _out_layers_env.strip():
        hierarchical_out_layers = [
            int(x) for x in _out_layers_env.replace(" ", "").split(",") if x != ""
        ]
    else:
        hierarchical_out_layers = list(_default_out_layers)
    shallow_layer_ids = [
        int(x) for x in cfgs_intermediate.get("shallow_layers", [5, 11])
    ]
    deep_layer_ids = [
        int(x) for x in cfgs_intermediate.get("deep_layers", [17, 23])
    ]
    shallow_layer_indices = [
        idx for idx, layer_id in enumerate(hierarchical_out_layers)
        if layer_id in shallow_layer_ids
    ]
    deep_layer_indices = [
        idx for idx, layer_id in enumerate(hierarchical_out_layers)
        if layer_id in deep_layer_ids
    ]
    intermediate_shallow_weight = float(
        cfgs_intermediate.get("shallow_weight", 1.5)
    )
    intermediate_deep_weight = float(
        cfgs_intermediate.get("deep_weight", 1.0)
    )
    intermediate_unseen_boundary_band = float(
        cfgs_intermediate.get("unseen_boundary_band", 0.10)
    )
    if intermediate_supervision_enabled and (
        len(shallow_layer_indices) == 0 or len(deep_layer_indices) == 0
    ):
        logger.warning(
            "Intermediate supervision enabled but could not map shallow=%s / "
            "deep=%s onto hierarchical_out_layers=%s; disabling reweighting.",
            shallow_layer_ids, deep_layer_ids, hierarchical_out_layers,
        )
        intermediate_supervision_enabled = False
    cfgs_completion = cfgs_loss.get("completion", {}) or {}
    completion_enabled = bool(cfgs_completion.get("enabled", False))
    completion_mask_source = str(cfgs_completion.get("mask_source", "rotation")).lower()
    if completion_mask_source not in ("rotation", "da3"):
        raise ValueError(
            f"loss.completion.mask_source must be 'rotation' or 'da3'; got {completion_mask_source!r}"
        )
    completion_loss_weight = float(cfgs_completion.get("weight", 0.5))
    completion_boundary_weight = float(cfgs_completion.get("boundary_weight", 0.5))
    completion_depth_tau = float(cfgs_completion.get("depth_tau", 0.20))
    completion_boundary_dilate = int(cfgs_completion.get("boundary_dilate", 1))
    completion_da3_model_id = str(
        cfgs_completion.get("da3_model_id", "depth-anything/DA3-GIANT-1.1")
    )
    completion_da3_depth_res = int(cfgs_completion.get("da3_depth_res", crop_size))
    cfgs_latent_patchgan = cfgs_loss.get("latent_patchgan", {}) or {}
    latent_patchgan_enabled = bool(cfgs_latent_patchgan.get("enabled", False))
    latent_patchgan_weight = float(cfgs_latent_patchgan.get("weight", 0.0))
    latent_patchgan_discriminator_weight = float(
        cfgs_latent_patchgan.get("discriminator_weight", 1.0)
    )
    latent_patchgan_lr = float(cfgs_latent_patchgan.get("lr", 1.0e-4))
    latent_patchgan_hidden_dim = int(cfgs_latent_patchgan.get("hidden_dim", 256))
    latent_patchgan_layers = int(cfgs_latent_patchgan.get("layers", 3))
    latent_patchgan_start_epoch = int(cfgs_latent_patchgan.get("start_epoch", 0))
    latent_patchgan_ramp_epochs = int(cfgs_latent_patchgan.get("ramp_epochs", 0))
    latent_patchgan_mask_source = str(
        cfgs_latent_patchgan.get("mask_source", "da3")
    ).lower()
    if latent_patchgan_mask_source not in ("da3", "rotation"):
        raise ValueError(
            "loss.latent_patchgan.mask_source must be 'da3' or 'rotation'; "
            f"got {latent_patchgan_mask_source!r}"
        )
    latent_patchgan_train_discriminator = bool(
        cfgs_latent_patchgan.get("train_discriminator", True)
    )
    latent_patchgan_input_layer_norm = bool(
        cfgs_latent_patchgan.get("input_layer_norm", False)
    )
    latent_patchgan_spectral_norm = bool(
        cfgs_latent_patchgan.get("spectral_norm", True)
    )
    latent_patchgan_beta1 = float(cfgs_latent_patchgan.get("beta1", 0.0))
    latent_patchgan_beta2 = float(cfgs_latent_patchgan.get("beta2", 0.99))
    da3_warp_enabled = warp_mode == "projective_da3"
    canvas_da3_boundary_enabled = (
        target_slot_mode == "canvas_first" and canvas_boundary_source == "da3"
    )
    if completion_enabled and not use_completion_head:
        logger.warning(
            "loss.completion.enabled=True but model.use_completion_head=False; "
            "enabling the masked region metrics but skipping completion-head loss."
        )
    latent_patchgan_da3_enabled = (
        latent_patchgan_enabled and latent_patchgan_mask_source == "da3"
    )
    if ((completion_enabled and completion_mask_source == "da3") or da3_warp_enabled or canvas_da3_boundary_enabled or latent_patchgan_da3_enabled) and not _DEPTH_TEACHER_AVAILABLE:
        raise ImportError(
            "DA3 depth usage requires src.training.common.depth_teacher."
        )
    # 1.4: when enabled, completion is blended into the *real* prediction path
    # rather than being a side-branch auxiliary loss only.
    #
    # pred_final = pred_raw * (1 - M) + comp_preds * M
    # where M = (reveal_mask + boundary_weight * boundary_mask).clamp(0, 1)
    #
    # pred_final is then used for: main loss, scheduled-sampling self-feed,
    # corrector input, cycle / rollout state.  This directly plugs the gap
    # where the completion head learns something useful but never fixes rollout.
    #
    # Requires: completion_enabled=True AND use_completion_head=True.
    # Off by default so all legacy configs remain bit-identical.
    completion_in_rollout = bool(cfgs_completion.get("in_rollout", False))
    # E6: residual alpha-blend for main-path completion instead of full replace.
    # pred_final = preds_next + alpha_now * M_birth * (comp_preds - preds_next)
    # alpha=0 => E5/E2 aux-only, alpha=1 => original full-replace (E3/E4).
    # alpha ramps from 0 to blend_alpha over alpha_ramp_epochs.
    completion_blend_alpha = float(cfgs_completion.get("blend_alpha", 0.0))
    completion_blend_alpha = float(os.environ.get("COMPLETION_BLEND_ALPHA", completion_blend_alpha))
    completion_blend_alpha_ramp_epochs = int(cfgs_completion.get("blend_alpha_ramp_epochs", 0))
    completion_blend_alpha_ramp_epochs = int(os.environ.get(
        "COMPLETION_BLEND_ALPHA_RAMP_EPOCHS", completion_blend_alpha_ramp_epochs
    ))
    completion_oov_blend_alpha_cfg = cfgs_completion.get("oov_blend_alpha", None)
    completion_boundary_blend_alpha_cfg = cfgs_completion.get("boundary_blend_alpha", None)
    _oov_blend_alpha_env = os.environ.get("COMPLETION_OOV_BLEND_ALPHA", None)
    _boundary_blend_alpha_env = os.environ.get("COMPLETION_BOUNDARY_BLEND_ALPHA", None)
    if _oov_blend_alpha_env is not None and str(_oov_blend_alpha_env).strip() == "":
        _oov_blend_alpha_env = None
    if _boundary_blend_alpha_env is not None and str(_boundary_blend_alpha_env).strip() == "":
        _boundary_blend_alpha_env = None
    completion_use_region_blend_alpha = (
        completion_oov_blend_alpha_cfg is not None
        or completion_boundary_blend_alpha_cfg is not None
        or _oov_blend_alpha_env is not None
        or _boundary_blend_alpha_env is not None
    )
    completion_oov_blend_alpha = float(
        _oov_blend_alpha_env
        if _oov_blend_alpha_env is not None
        else (completion_oov_blend_alpha_cfg if completion_oov_blend_alpha_cfg is not None else completion_blend_alpha)
    )
    completion_boundary_blend_alpha = float(
        _boundary_blend_alpha_env
        if _boundary_blend_alpha_env is not None
        else (completion_boundary_blend_alpha_cfg if completion_boundary_blend_alpha_cfg is not None else 0.25 * completion_blend_alpha)
    )
    # E7: delay self-feed of pred_final (use preds_next for rollout) for the
    # first N epochs, then switch to pred_final. 0 = no delay (E3/E4 behaviour).
    completion_selffeed_delay_epochs = int(cfgs_completion.get("selffeed_delay_epochs", 0))
    completion_selffeed_delay_epochs = int(os.environ.get(
        "COMPLETION_SELFFEED_DELAY_EPOCHS", completion_selffeed_delay_epochs
    ))
    if completion_in_rollout and not (completion_enabled and use_completion_head):
        logger.warning(
            "loss.completion.in_rollout=True but completion is not fully enabled "
            "(completion_enabled=%s, use_completion_head=%s); forcing in_rollout=False.",
            completion_enabled, use_completion_head,
        )
        completion_in_rollout = False
    # Run B — residual target: predict z_target - warp(z_last, target)
    # instead of full z_target.  Requires warp_mode != "off" and a valid
    # depth_probe_checkpoint so the warped context is available.
    residual_target = bool(cfgs_loss.get("residual_target", False))
    if residual_target and warp_mode == "off":
        logger.warning(
            "residual_target=True but warp_mode='off': warped_context_raw will "
            "always be None so residual_target is a no-op. Forcing residual_target=False."
        )
        residual_target = False
    residual_target_reconstruct = bool(
        cfgs_loss.get("residual_target_reconstruct", residual_target)
    )
    residual_full_loss_weight = float(
        cfgs_loss.get("residual_full_loss_weight", 0.0)
    )
    if not residual_target:
        residual_target_reconstruct = False
        residual_full_loss_weight = 0.0

    def _reconstruct_residual_prediction(preds, warped_context_raw):
        if (
            residual_target
            and residual_target_reconstruct
            and warped_context_raw is not None
        ):
            return preds + warped_context_raw.to(
                device=preds.device,
                dtype=preds.dtype,
            )
        return preds

    rollout_train_steps = max(1, int(cfgs_loss.get("rollout_train_steps", 1)))
    rollout_loss_decay = float(cfgs_loss.get("rollout_loss_decay", 0.5))

    # Scheduled sampling (probabilistic teacher forcing) for rollout context.
    # Controls how the predictor's rollout context is built between steps k
    # and k+1 during training: with probability ``scheduled_sampling_prob`` we
    # self-feed (append the predictor's own output, i.e. the current default
    # two-step behaviour), and with probability ``1 - p`` we teacher-force
    # (append the GT target latent instead). Defaults to ``1.0`` so legacy
    # configs (and the existing ``two_step_rollout`` experiment) are
    # bit-identical to before.
    #
    # When ``scheduled_sampling_warmup_epochs > 0``, p ramps linearly from 0
    # to ``scheduled_sampling_prob`` over the first N epochs, then stays at
    # the target for the rest of training. A warmup of 0 means use the target
    # p from step 0 (equivalent to fixed two_step_rollout when p=1.0).
    #
    # Interpretation: p=1.0 is pure self-feeding (current); p=0.0 is pure
    # teacher forcing at every rollout step; p in between mixes the two.
    # Reducing p typically improves single-step metrics at the cost of the
    # distribution-shift training signal, so low p is useful in combination
    # with larger ``rollout_train_steps`` (k=3,4) where a fully self-fed chain
    # would otherwise blow up the training loss early on.
    scheduled_sampling_prob = float(
        cfgs_loss.get("scheduled_sampling_prob", 1.0)
    )
    scheduled_sampling_prob = min(max(scheduled_sampling_prob, 0.0), 1.0)
    scheduled_sampling_warmup_epochs = int(
        cfgs_loss.get("scheduled_sampling_warmup_epochs", 0)
    )
    if rollout_train_steps > 1 or scheduled_sampling_prob < 1.0:
        logger.info(
            "Rollout training: rollout_train_steps=%d rollout_loss_decay=%.3f "
            "scheduled_sampling_prob=%.3f warmup_epochs=%d",
            rollout_train_steps,
            rollout_loss_decay,
            scheduled_sampling_prob,
            scheduled_sampling_warmup_epochs,
        )

    # VGGT-World / JEPA-WMS-style soft self-conditioning. Scheduled sampling
    # decides whether a rollout step uses the model prediction or the teacher
    # target as the next context slot; this optional curriculum softens the
    # self-fed branch itself by blending the predicted context with the GT
    # latent:
    #   c_next = (1 - lambda) * z_gt + lambda * z_pred
    # ``mode=linear`` uses one deterministic epoch-level lambda. ``mode=beta``
    # samples a per-sample lambda from a broadening Beta distribution, giving
    # the v1 rollout trainer a continuous context-quality spectrum analogous to
    # the v2 flow-forcing context-noise curriculum.
    cfgs_context_mix = cfgs_loss.get("rollout_context_mix", {}) or {}
    rollout_context_mix_enabled = bool(cfgs_context_mix.get("enabled", False))
    rollout_context_mix_mode = str(cfgs_context_mix.get("mode", "linear")).lower()
    if rollout_context_mix_mode not in {"linear", "beta"}:
        logger.warning(
            "Unknown rollout_context_mix.mode=%r; falling back to 'linear'",
            rollout_context_mix_mode,
        )
        rollout_context_mix_mode = "linear"
    rollout_context_mix_max_lambda = min(
        max(float(cfgs_context_mix.get("max_lambda", 1.0)), 0.0),
        1.0,
    )
    rollout_context_mix_start_epoch = int(cfgs_context_mix.get("start_epoch", 0))
    rollout_context_mix_ramp_end_epoch = int(
        cfgs_context_mix.get("ramp_end_epoch", max(1, scheduled_sampling_warmup_epochs))
    )
    rollout_context_mix_beta_start_a = max(float(cfgs_context_mix.get("beta_start_a", 2.0)), 1.0e-4)
    rollout_context_mix_beta_start_b = max(float(cfgs_context_mix.get("beta_start_b", 8.0)), 1.0e-4)
    rollout_context_mix_beta_end_a = max(float(cfgs_context_mix.get("beta_end_a", 2.0)), 1.0e-4)
    rollout_context_mix_beta_end_b = max(float(cfgs_context_mix.get("beta_end_b", 2.0)), 1.0e-4)
    if rollout_context_mix_enabled:
        logger.info(
            "Rollout context mix: enabled mode=%s max_lambda=%.3f start_epoch=%d "
            "ramp_end_epoch=%d beta_start=(%.3f, %.3f) beta_end=(%.3f, %.3f)",
            rollout_context_mix_mode,
            rollout_context_mix_max_lambda,
            rollout_context_mix_start_epoch,
            rollout_context_mix_ramp_end_epoch,
            rollout_context_mix_beta_start_a,
            rollout_context_mix_beta_start_b,
            rollout_context_mix_beta_end_a,
            rollout_context_mix_beta_end_b,
        )

    # A3: Gaussian perturbation applied to self-fed predicted latents during
    # rollout training. Std is expressed as a *fraction* of the per-channel
    # GT target-latent std — so 0.05 means "add noise at 5% of the target's
    # per-channel scale". 0.0 (default) is the exact legacy two_step_rollout
    # behaviour (no noise). Only active under self-feeding; teacher-forced
    # context slots are passed through untouched.
    rollout_noise_std = float(cfgs_loss.get("rollout_noise_std", 0.0))

    # SIGReg (LeWM 2026 App. A) applied to rolled-forward predicted latents
    # against the GT latent distribution. Unlike FSQ (output-side CE) this
    # shapes the distribution of latents that will be self-fed into the
    # next rollout step — directly targets the closed-loop input-
    # distribution mismatch. weight=0.0 disables; 0.02 is conservative.
    sigreg_weight = float(cfgs_loss.get("sigreg_weight", 0.0))
    sigreg_n_directions = int(cfgs_loss.get("sigreg_n_directions", 64))
    if rollout_noise_std > 0.0 or sigreg_weight > 0.0:
        logger.info(
            "Input-distribution shaping: rollout_noise_std=%.4f "
            "sigreg_weight=%.4f sigreg_n_directions=%d",
            rollout_noise_std, sigreg_weight, sigreg_n_directions,
        )

    # Motion-magnitude-weighted L1. When enabled, each rollout step's L1 loss
    # is reweighted by the per-sample action magnitude so high-motion samples
    # receive proportionally more gradient. Normalized to mean=1 per step so
    # overall loss scale is preserved when motion is uniform. Targets the
    # ``T-hi`` bin regression observed in the motion/rollout ablation table.
    motion_weighted = bool(cfgs_loss.get("motion_weighted", False))
    motion_weight_alpha = float(cfgs_loss.get("motion_weight_alpha", 1.0))
    motion_weight_mode = str(cfgs_loss.get("motion_weight_mode", "translation"))
    if motion_weighted and motion_weight_alpha <= 0.0:
        logger.warning(
            "motion_weighted=True but motion_weight_alpha=%.3f <= 0; the loss "
            "will be identical to the unweighted baseline.",
            motion_weight_alpha,
        )
    rollout_train_steps = min(rollout_train_steps, max_valid_ctx_tubelets)
    max_rollout_ctx_tubelets = max(1, total_tubelets - rollout_train_steps)
    if rollout_train_steps > 1 and max_ctx_tubelets > max_rollout_ctx_tubelets:
        logger.warning(
            f"rollout_train_steps={rollout_train_steps} with total_tubelets={total_tubelets} "
            f"caps effective training context to at most {max_rollout_ctx_tubelets} tubelets."
        )

    # Phase R — reversibility cycle loss config (TCR design section 3).
    # Disabled by default. The pilot uses K=1, predicted-seed, no visibility
    # mask; see ``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` section 3.7 for
    # the full key surface.
    cfgs_cycle = cfgs_loss.get("cycle", {}) or {}
    cycle_enabled = bool(cfgs_cycle.get("enabled", False))
    cycle_weight = float(cfgs_cycle.get("weight", 0.02))
    cycle_horizon = int(cfgs_cycle.get("horizon", 1))
    cycle_start_epoch = int(cfgs_cycle.get("start_epoch", 5))
    cycle_ramp_end_epoch = int(cfgs_cycle.get("ramp_end_epoch", 15))
    cycle_seed_mode = str(cfgs_cycle.get("seed", "predicted")).lower()
    if cycle_seed_mode not in ("predicted", "teacher_forced"):
        raise ValueError(
            f"loss.cycle.seed must be 'predicted' or 'teacher_forced'; "
            f"got {cycle_seed_mode!r}"
        )
    cycle_visibility_mask = str(cfgs_cycle.get("visibility_mask", "none")).lower()
    if cycle_visibility_mask not in ("none", "adaptive"):
        raise ValueError(
            f"loss.cycle.visibility_mask must be 'none' or 'adaptive'; "
            f"got {cycle_visibility_mask!r}"
        )
    cycle_detach_forward = bool(cfgs_cycle.get("detach_forward", False))
    if cycle_enabled and cycle_horizon != 1:
        raise NotImplementedError(
            "Phase R pilot supports horizon=1 only. Multi-step cycles (R3) are "
            "deferred; see docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md section 3.3."
        )
    if cycle_enabled:
        logger.info(
            "Phase R reversibility cycle: enabled K=%d weight=%.4f seed=%s "
            "visibility_mask=%s detach_forward=%s start_epoch=%d ramp_end_epoch=%d",
            cycle_horizon, cycle_weight, cycle_seed_mode,
            cycle_visibility_mask, cycle_detach_forward,
            cycle_start_epoch, cycle_ramp_end_epoch,
        )
    else:
        logger.info("Phase R reversibility cycle: disabled.")

    # Phase R' — SE(3) group-composition consistency loss config.
    # Disabled by default. The pilot uses teacher-seeded composition with
    # supervised_anchor=true and path_path_weight=0; see
    # ``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` for the design rationale.
    cfgs_composition = cfgs_loss.get("composition", {}) or {}
    composition_enabled = bool(cfgs_composition.get("enabled", False))
    composition_weight = float(cfgs_composition.get("weight", 0.02))
    composition_start_epoch = int(cfgs_composition.get("start_epoch", 5))
    composition_ramp_end_epoch = int(cfgs_composition.get("ramp_end_epoch", 15))
    composition_seed_mode = str(cfgs_composition.get("seed", "teacher")).lower()
    if composition_seed_mode not in ("teacher", "predicted"):
        raise ValueError(
            f"loss.composition.seed must be 'teacher' or 'predicted'; "
            f"got {composition_seed_mode!r}"
        )
    composition_supervised_anchor = bool(
        cfgs_composition.get("supervised_anchor", True)
    )
    composition_path_path_weight = float(
        cfgs_composition.get("path_path_weight", 0.0)
    )
    composition_detach_rhs = bool(cfgs_composition.get("detach_rhs", True))
    composition_max_pairs = int(cfgs_composition.get("max_pairs_per_batch", 1))
    if composition_enabled and composition_max_pairs != 1:
        raise NotImplementedError(
            "Phase R' pilot supports max_pairs_per_batch=1 only; multi-pair "
            "composition is deferred. Got max_pairs_per_batch="
            f"{composition_max_pairs}."
        )
    if composition_enabled and not (
        composition_supervised_anchor or composition_path_path_weight > 0.0
    ):
        raise ValueError(
            "loss.composition is enabled but both supervised_anchor=False "
            "and path_path_weight=0 — the loss has no signal. Enable at "
            "least one of the two terms."
        )
    if composition_enabled:
        logger.info(
            "Phase R' group-composition: enabled weight=%.4f seed=%s "
            "supervised_anchor=%s path_path_weight=%.4f detach_rhs=%s "
            "start_epoch=%d ramp_end_epoch=%d",
            composition_weight, composition_seed_mode,
            composition_supervised_anchor, composition_path_path_weight,
            composition_detach_rhs,
            composition_start_epoch, composition_ramp_end_epoch,
        )
    else:
        logger.info("Phase R' group-composition: disabled.")

    # -- OPTIMIZATION
    cfgs_opt = args.get("optimization")
    ipe = cfgs_opt.get("ipe", None)
    wd = float(cfgs_opt.get("weight_decay"))
    final_wd = float(cfgs_opt.get("final_weight_decay"))
    num_epochs = cfgs_opt.get("epochs")
    anneal = cfgs_opt.get("anneal")
    warmup = cfgs_opt.get("warmup")
    start_lr = cfgs_opt.get("start_lr")
    lr = cfgs_opt.get("lr")
    final_lr = cfgs_opt.get("final_lr")
    enc_lr_scale = cfgs_opt.get("enc_lr_scale", 1.0)
    betas = cfgs_opt.get("betas", (0.9, 0.999))
    eps = cfgs_opt.get("eps", 1.0e-8)

    # ----------------------------------------------------------------------- #

    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.benchmark = True
    try:
        mp.set_start_method("spawn")
    except Exception:
        pass

    world_size, rank = init_distributed()
    logger.info(f"Initialized (rank/world-size) {rank}/{world_size}")
    # Silence non-rank-0 loggers to cut the duplicate per-rank log spam
    # (model repr, init banners, dataset indexing, stats). Set
    # VJEPA_LOG_ALL_RANKS=1 to restore full per-rank logging.
    if rank != 0 and os.environ.get("VJEPA_LOG_ALL_RANKS", "0") != "1":
        logging.getLogger().setLevel(logging.WARNING)

    if not torch.cuda.is_available():
        device = torch.device("cpu")
    else:
        device = torch.device("cuda:0")
        torch.cuda.set_device(device)

    os.makedirs(scratch_folder, exist_ok=True)
    log_file = os.path.join(folder, f"log_r{rank}.csv")
    final_path = os.path.join(scratch_folder, "final.pt")
    checkpoint_stage_dir = os.environ.get("CHECKPOINT_STAGE_DIR")
    if checkpoint_stage_dir is None:
        checkpoint_stage_root = "/mnt/resource" if os.path.isdir("/mnt/resource") else os.environ.get("TMPDIR", "/tmp")
        checkpoint_stage_dir = os.path.join(checkpoint_stage_root, "vjepa_camera_ckpt_stage")
    os.makedirs(checkpoint_stage_dir, exist_ok=True)
    if r_file is not None:
        resume_path = os.path.join(scratch_folder, r_file)
        if not os.path.exists(resume_path):
            resume_path = None
    else:
        epoch_checkpoints = sorted(glob.glob(os.path.join(scratch_folder, "e*.pt")))
        resume_path = final_path if os.path.exists(final_path) else (epoch_checkpoints[-1] if epoch_checkpoints else None)

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_pred"),
        ("%.5f", "loss_ctx"),
        ("%.5f", "loss_step1"),
        ("%.5f", "loss_step2"),
        ("%d", "iter-time(ms)"),
        ("%d", "gpu-time(ms)"),
        ("%d", "dataload-time(ms)"),
        mode="+a",
    )

    # -- init model
    encoder, predictor = init_video_model(
        device=device,
        patch_size=patch_size,
        max_num_frames=seq_len * tubelet_size,   # total frames for mask pre-compute
        tubelet_size=tubelet_size,
        model_name=model_name,
        crop_size=crop_size,
        pred_depth=pred_depth,
        pred_num_heads=pred_num_heads,
        pred_embed_dim=pred_embed_dim,
        n_hierarchical_layers=n_hierarchical_layers,
        out_embed_dim=None,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        use_pred_silu=use_pred_silu,
        wide_silu=wide_silu,
        use_rope=use_rope,
        pred_is_frame_causal=pred_is_frame_causal,
        use_activation_checkpointing=use_activation_checkpointing,
        state_dim=state_dim,
        action_dim=action_dim,
        intrinsics_dim=intrinsics_dim,
        use_intrinsics=use_intrinsics,
        predict_all=predict_all,
        use_ray_pe=use_ray_pe,
        ray_pe_dim=ray_pe_dim,
        ray_pe_hidden=ray_pe_hidden,
        ray_pe_mode=ray_pe_mode,
        ray_visibility_features=ray_visibility_features,
        pose_conditioning_mode=pose_conditioning_mode,
        action_token_mode=action_token_mode,
        use_delta_head=use_delta_head,
        use_residual_head=use_residual_head,
        residual_head_depth=residual_head_depth,
        residual_head_ratio=residual_head_ratio,
        use_completion_head=use_completion_head,
        completion_head_depth=completion_head_depth,
        completion_head_ratio=completion_head_ratio,
        use_four_layer_birth_head=use_four_layer_birth_head,
        completion_head_hidden=completion_head_hidden,
        use_fsq_head=use_fsq_head,
        fsq_total_code_axes=fsq_total_code_axes,
        fsq_levels=fsq_levels,
        warp_context_latents=warp_context_latents,
        warp_context_padding_mode=warp_context_padding_mode,
        warp_mode=warp_mode,
        depth_probe_checkpoint=depth_probe_checkpoint,
        target_slot_mode=target_slot_mode,
        canvas_beta_init_valid=canvas_beta_init_valid,
        canvas_beta_init_boundary=canvas_beta_init_boundary,
        canvas_beta_init_oov=canvas_beta_init_oov,
        canvas_use_mask_feat_embed=canvas_use_mask_feat_embed,
        canvas_use_token_type_embed=canvas_use_token_type_embed,
        canvas_use_target_step_embed=canvas_use_target_step_embed,
        canvas_warp_mode=canvas_warp_mode,
        canvas_projector_gamma_init=canvas_projector_gamma_init,
        canvas_norm_clip_ratio=canvas_norm_clip_ratio,
        canvas_block_mask_enabled=canvas_block_mask_enabled,
        canvas_block_mask_min_rects=canvas_block_mask_min_rects,
        canvas_block_mask_max_rects=canvas_block_mask_max_rects,
        canvas_block_mask_min_frac=canvas_block_mask_min_frac,
        canvas_block_mask_max_frac=canvas_block_mask_max_frac,
        camera_ucpe_enabled=camera_ucpe_enabled,
        camera_ucpe_apply_layers=camera_ucpe_apply_layers,
        camera_ucpe_gamma_init=camera_ucpe_gamma_init,
        predictor_adaln_enabled=predictor_adaln_enabled,
        predictor_adaln_hidden=predictor_adaln_hidden,
        correspondence_bias_enabled=correspondence_bias_enabled,
        correspondence_bias_mode=correspondence_bias_mode,
        correspondence_bias_sigma_tokens=correspondence_bias_sigma_tokens,
        correspondence_bias_lambda_init=correspondence_bias_lambda_init,
        correspondence_bias_learnable=correspondence_bias_learnable,
        correspondence_bias_apply_layers=correspondence_bias_apply_layers,
        encoder_backbone=encoder_backbone,
        encoder_freeze=encoder_freeze,
    )
    target_encoder = copy.deepcopy(encoder)

    # Move the FSQ tokenizer to the training device (frozen, so no optimizer).
    if fsq_module is not None:
        fsq_module = fsq_module.to(device)

    # ------------------------------------------------------------------ #
    # Phase C — latent corrector (TCR design §5)
    # ------------------------------------------------------------------ #
    cfgs_corrector = cfgs_model.get("corrector", {}) or {}
    corrector_enabled = bool(cfgs_corrector.get("enabled", False))
    corrector_mode = str(cfgs_corrector.get("mode", "joint")).lower()  # "frozen_predictor" | "joint"
    if corrector_mode not in ("frozen_predictor", "joint"):
        raise ValueError(
            f"model.corrector.mode must be 'frozen_predictor' or 'joint'; got {corrector_mode!r}"
        )
    corrector_denoise_weight = float(cfgs_corrector.get("denoise_weight", 0.3))
    corrector_noise_std = float(cfgs_corrector.get("denoise_noise_std", rollout_noise_std))
    corrector = None
    if corrector_enabled:
        # Determine corrector input width: predictor output is
        # n_hierarchical_layers * out_embed_dim concatenated. The predictor's
        # ``out_embed_dim`` defaults to ``embed_dim`` when not set; we read it
        # from the live module if available, falling back to ``embed_dim``.
        _pred_unwrapped = predictor
        _corr_embed_dim = int(getattr(_pred_unwrapped, "out_embed_dim", None) or embed_dim)
        # Cap step embedding table size by the maximum rollout horizon we
        # might exercise in eval (rollout_train_steps for training, plus
        # any horizon used by camera_rollout_pca eval).
        _corr_max_steps = int(
            cfgs_corrector.get("max_rollout_steps", max(32, rollout_train_steps + 4))
        )
        corrector = build_latent_corrector(
            n_hierarchical_layers=n_hierarchical_layers,
            embed_dim=_corr_embed_dim,
            cfg={**cfgs_corrector, "max_rollout_steps": _corr_max_steps},
        ).to(device)
        n_corr_params = sum(p.numel() for p in corrector.parameters())
        logger.info(
            "Phase C latent corrector: enabled mode=%s n_layers=%d embed_dim=%d "
            "params=%d denoise_weight=%.3f noise_std=%.4f",
            corrector_mode, n_hierarchical_layers, _corr_embed_dim,
            n_corr_params, corrector_denoise_weight, corrector_noise_std,
        )
        if corrector_mode == "frozen_predictor":
            # Freeze encoder + predictor so only the corrector trains. We do
            # not freeze target_encoder — it is already frozen at line 866
            # below. We also keep encoder/predictor in eval() to disable
            # dropout etc.; this is set per-iteration in the training loop.
            for _p in encoder.parameters():
                _p.requires_grad = False
            for _p in predictor.parameters():
                _p.requires_grad = False
            logger.info(
                "Phase C: corrector_mode=frozen_predictor — encoder + predictor parameters frozen."
            )
    else:
        logger.info("Phase C latent corrector: disabled.")

    latent_patchgan = None
    if latent_patchgan_enabled:
        _pred_for_patchgan_dims = predictor.module if hasattr(predictor, "module") else predictor
        _patchgan_layer_dim = getattr(_pred_for_patchgan_dims, "out_embed_dim", None)
        if _patchgan_layer_dim is None:
            raise RuntimeError(
                "Latent PatchGAN needs predictor.out_embed_dim to infer the "
                "concatenated hierarchical token width."
            )
        latent_patchgan_in_dim = int(n_hierarchical_layers) * int(_patchgan_layer_dim)
        latent_patchgan = LatentPatchDiscriminator(
            in_dim=latent_patchgan_in_dim,
            hidden_dim=latent_patchgan_hidden_dim,
            n_layers=latent_patchgan_layers,
            input_layer_norm=latent_patchgan_input_layer_norm,
            spectral_norm=latent_patchgan_spectral_norm,
        ).to(device)
        if not latent_patchgan_train_discriminator:
            latent_patchgan.eval()
            _set_module_requires_grad(latent_patchgan, False)
        n_patchgan_params = sum(p.numel() for p in latent_patchgan.parameters())
        logger.info(
            "Latent PatchGAN discriminator params=%d token_dim=%d grid=%dx%d.",
            n_patchgan_params,
            latent_patchgan_in_dim,
            crop_size // patch_size,
            crop_size // patch_size,
        )

    completion_depth_teacher = None
    completion_depth_processor = None
    if completion_enabled:
        logger.info(
            "Completion route: enabled=%s head=%s mask_source=%s weight=%.3f "
            "boundary_weight=%.3f depth_tau=%.3f boundary_dilate=%d",
            completion_enabled,
            use_completion_head,
            completion_mask_source,
            completion_loss_weight,
            completion_boundary_weight,
            completion_depth_tau,
            completion_boundary_dilate,
        )
    if latent_patchgan_enabled:
        logger.info(
            "Latent PatchGAN: enabled weight=%.4f d_weight=%.4f mask_source=%s "
            "lr=%.2e hidden=%d layers=%d start_epoch=%d ramp_epochs=%d "
            "train_D=%s input_ln=%s spectral_norm=%s",
            latent_patchgan_weight,
            latent_patchgan_discriminator_weight,
            latent_patchgan_mask_source,
            latent_patchgan_lr,
            latent_patchgan_hidden_dim,
            latent_patchgan_layers,
            latent_patchgan_start_epoch,
            latent_patchgan_ramp_epochs,
            latent_patchgan_train_discriminator,
            latent_patchgan_input_layer_norm,
            latent_patchgan_spectral_norm,
        )
    else:
        logger.info("Latent PatchGAN: disabled.")
    if (completion_enabled and completion_mask_source == "da3") or da3_warp_enabled or canvas_da3_boundary_enabled or latent_patchgan_da3_enabled:
        if load_depth_teacher is None:
            raise RuntimeError("DA3 depth requested but depth teacher loader is unavailable.")
        completion_depth_processor, completion_depth_teacher = load_depth_teacher(
            device=device,
            model_id=completion_da3_model_id,
        )
        logger.info(
            "DA3 depth teacher: model=%s depth_res=%d completion_masks=%s warp=%s canvas_boundary=%s patchgan=%s",
            completion_da3_model_id,
            completion_da3_depth_res,
            completion_enabled and completion_mask_source == "da3",
            da3_warp_enabled,
            canvas_da3_boundary_enabled,
            latent_patchgan_da3_enabled,
        )

    if compile_model:
        logger.info("Compiling encoder, target_encoder, and predictor.")
        torch._dynamo.config.optimize_ddp = False
        encoder = torch.compile(encoder)
        target_encoder = torch.compile(target_encoder)
        predictor = torch.compile(predictor)

    # -- init data
    # -- init eval loader for PCA visualisation (test split, no DDP sampler)
    pca_vis_dir = os.path.join(folder, "pca_vis")
    os.makedirs(pca_vis_dir, exist_ok=True)
    fixed_stride_eval_loaders = {}
    try:
        gwm_root = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "..", ".."))
        if gwm_root not in sys.path:
            sys.path.insert(0, gwm_root)
        from src.core.dataset import RE10KLazySceneDataset
        from src.training.multiscene.re10k_sequence_dataset import RE10KSequenceDataset, collate_re10k_sequences
        _eval_lazy = RE10KLazySceneDataset(
            root=data_root, stage="test",
            image_size=crop_size,
            manifest_cache_dir=manifest_cache_dir,
            fixed_manifest_path=eval_fixed_manifest_path,
            local_chunk_cache_dir=local_chunk_cache_dir,
            local_chunk_cache_limit_gb=local_chunk_cache_limit_gb,
        )
        def _make_eval_loader(eval_stride_override=None):
            if eval_stride_override is None:
                eval_stride = stride
                eval_min_stride = min_stride
                eval_stride_values = stride_values
            else:
                eval_stride = int(eval_stride_override)
                eval_min_stride = eval_stride
                eval_stride_values = [eval_stride]
            _eval_ds = RE10KSequenceDataset(
                lazy_dataset=_eval_lazy, seq_len=seq_len, stride=eval_stride,
                min_stride=eval_min_stride, stride_values=eval_stride_values, image_size=crop_size,
            )
            _loader = DataLoader(
                _eval_ds, batch_size=1, shuffle=False,
                num_workers=0, pin_memory=False,
                collate_fn=collate_re10k_sequences, drop_last=False,
            )
            return _loader, len(_eval_ds)

        pca_loader, _eval_ds_len = _make_eval_loader()
        fixed_stride_eval_loaders = {
            "small": (eval_stride_small, _make_eval_loader(eval_stride_small)[0]),
            "medium": (eval_stride_medium, _make_eval_loader(eval_stride_medium)[0]),
            "large": (eval_stride_large, _make_eval_loader(eval_stride_large)[0]),
        }
        logger.info(f"PCA eval loader: {_eval_ds_len} test scenes")
        logger.info(
            f"Fixed stride eval loaders: small={eval_stride_small}, medium={eval_stride_medium}, large={eval_stride_large}"
        )
    except Exception as _e:
        pca_loader = None
        fixed_stride_eval_loaders = {}
        logger.warning(f"PCA eval loader unavailable: {_e}")

    unsupervised_loader, unsupervised_sampler = _init_re10k_loader(
        data_root=data_root,
        seq_len=seq_len,
        stride=stride,
        min_stride=min_stride,
        stride_values=stride_values,
        image_size=crop_size,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_mem=pin_mem,
        persistent_workers=persistent_workers,
        world_size=world_size,
        rank=rank,
        manifest_cache_dir=manifest_cache_dir,
        fixed_manifest_path=fixed_manifest_path,
        local_chunk_cache_dir=local_chunk_cache_dir,
        local_chunk_cache_limit_gb=local_chunk_cache_limit_gb,
    )
    _dlen = len(unsupervised_loader)
    if ipe is None:
        ipe = _dlen
    logger.info(f"iterations per epoch/dataset length: {ipe}/{_dlen}")

    # Freeze context encoder before optimizer construction so its parameters
    # are excluded from AdamW param groups entirely (no moment allocation,
    # no zero-gradient step overhead). The enc_lr_scale<=0 path previously
    # did this after init_opt, which wasted ~2.4 GB optimizer state on ViT-L
    # (8.8 GB on ViT-G) and added a small per-step cost.
    context_encoder_frozen = float(enc_lr_scale) <= 0.0
    if context_encoder_frozen:
        for p in encoder.parameters():
            p.requires_grad = False
        encoder.eval()
        logger.info(
            "Context encoder frozen (enc_lr_scale=%.3f <= 0): "
            "parameters set requires_grad=False and module.eval() called.",
            float(enc_lr_scale),
        )

    # -- init optimizer
    # Phase C: corrector is appended as an additional param group when
    # enabled. In ``frozen_predictor`` mode encoder + predictor params have
    # already had ``requires_grad=False`` set above, so ``init_opt`` will
    # skip them via the requires_grad filter.
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
        corrector=corrector,
        wd=wd,
        final_wd=final_wd,
        start_lr=start_lr,
        ref_lr=lr,
        final_lr=final_lr,
        enc_lr_scale=enc_lr_scale,
        iterations_per_epoch=ipe,
        anneal=anneal,
        warmup=warmup,
        num_epochs=num_epochs,
        mixed_precision=mixed_precision,
        betas=betas,
        eps=eps,
    )

    use_ddp = world_size > 1 and dist.is_available() and dist.is_initialized()

    def _has_trainable_params(module):
        """DDP raises ``RuntimeError: DistributedDataParallel is not needed when
        a module doesn't have any parameter that requires a gradient.`` In
        Phase C frozen-predictor mode (TCR §5) the encoder + predictor are
        fully frozen, so we must skip DDP wrapping for them and broadcast
        their parameters once at startup instead. ``target_encoder`` has
        always been frozen (set at module init) and is similarly handled."""
        return any(p.requires_grad for p in module.parameters())

    if use_ddp:
        if _has_trainable_params(encoder):
            encoder = DistributedDataParallel(encoder, static_graph=True)
            logger.info("Wrapped encoder with DistributedDataParallel.")
        else:
            logger.info(
                "encoder has no trainable parameters; skipping DDP wrap "
                "(Phase C frozen_predictor or enc_lr_scale<=0)."
            )
        if _has_trainable_params(predictor):
            # The completion head is now called unconditionally per rollout
            # step (loss-weighting is the only data-dependent piece, which
            # autograd happily folds into a zero-gradient contribution when
            # no boundary/disoccluded tokens are present). That keeps the
            # graph static across iterations, so we can use ``static_graph=True``
            # — the only DDP mode that correctly handles a parameter being
            # used multiple times per backward (multi-step rollout).
            predictor = DistributedDataParallel(predictor, static_graph=True)
            logger.info("Wrapped predictor with DistributedDataParallel (static_graph=True).")
        else:
            logger.info(
                "predictor has no trainable parameters; skipping DDP wrap "
                "(Phase C frozen_predictor mode)."
            )
        # target_encoder is always frozen (EMA target). Wrapping it works
        # because DDP only refuses when the *module* has zero trainable
        # params at wrap time. Older code wrapped it unconditionally; keep
        # that behavior only when there are trainable params, otherwise
        # leave unwrapped.
        if _has_trainable_params(target_encoder):
            target_encoder = DistributedDataParallel(target_encoder)
        if corrector is not None and _has_trainable_params(corrector):
            # Corrector has dynamic per-step rollout structure (called K times
            # per iteration with shared params); ``static_graph=False`` is
            # safer until we verify static-graph compatibility.
            corrector = DistributedDataParallel(corrector)
            logger.info("Wrapped corrector with DistributedDataParallel.")
        if latent_patchgan is not None and _has_trainable_params(latent_patchgan):
            latent_patchgan = DistributedDataParallel(latent_patchgan)
            logger.info("Wrapped latent PatchGAN discriminator with DistributedDataParallel.")
    else:
        if world_size > 1:
            logger.warning(
                "world_size=%s but torch.distributed is not initialized; proceeding without DDP.",
                world_size,
            )
        else:
            logger.info("Running single-process training without DistributedDataParallel.")
    for p in target_encoder.parameters():
        p.requires_grad = False
    predictor_ref = predictor.module if hasattr(predictor, "module") else predictor
    latent_patchgan_optimizer = None
    if latent_patchgan is not None and latent_patchgan_train_discriminator:
        latent_patchgan_optimizer = torch.optim.AdamW(
            [p for p in latent_patchgan.parameters() if p.requires_grad],
            lr=latent_patchgan_lr,
            betas=(latent_patchgan_beta1, latent_patchgan_beta2),
            weight_decay=0.0,
        )

    logger.info(
        "Loss config: normalize_reps=%s, per_layer_balance=%s, loss_exp=%.3f, "
        "rollout_train_steps=%d, motion_weighted=%s (mode=%s, alpha=%.2f), "
        "intermediate_supervision=%s, residual_target=%s "
        "(reconstruct=%s full_w=%.3f).",
        normalize_reps, per_layer_balance, float(loss_exp), rollout_train_steps,
        motion_weighted, motion_weight_mode, motion_weight_alpha,
        intermediate_supervision_enabled,
        residual_target,
        residual_target_reconstruct,
        residual_full_loss_weight,
    )
    if intermediate_supervision_enabled:
        logger.info(
            "Intermediate supervision: out_layers=%s shallow(ids=%s idx=%s x%.2f) "
            "deep(ids=%s idx=%s x%.2f) unseen_boundary_band=%.3f",
            hierarchical_out_layers,
            shallow_layer_ids,
            shallow_layer_indices,
            intermediate_shallow_weight,
            deep_layer_ids,
            deep_layer_indices,
            intermediate_deep_weight,
            intermediate_unseen_boundary_band,
        )

    # -- load pretrained encoder weights
    # Adapter backbones load their own pretrained weights, so the V-JEPA
    # pretrain checkpoint must not be re-applied to the encoder. We still
    # let ``load_pretrained`` run so it can warm-start the predictor when
    # ``load_predictor=True``; just disable the encoder side.
    _adapter_backbones = {"mast3r", "cradio"}
    _load_encoder_effective = load_encoder and (encoder_backbone not in _adapter_backbones)
    if encoder_backbone in _adapter_backbones and load_encoder:
        logger.info(
            f"encoder_backbone={encoder_backbone}: skipping load_pretrained on "
            "encoder/target_encoder (adapter weights are already loaded)."
        )
    if _load_encoder_effective or load_predictor:
        encoder, predictor, target_encoder = load_pretrained(
            r_path=p_file,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            context_encoder_key=context_encoder_key,
            target_encoder_key=target_encoder_key,
            load_predictor=load_predictor,
            load_encoder=_load_encoder_effective,
        )
    else:
        logger.info(
            "Skipping load_pretrained entirely: no encoder/predictor weights "
            "requested for this backbone."
        )

    def _remove_file(path):
        try:
            os.remove(path)
        except FileNotFoundError:
            pass
        except Exception as e:
            logger.warning(f"Failed to remove {path}: {e}")

    def _remove_tree(path):
        if not path:
            return
        try:
            shutil.rmtree(path)
        except FileNotFoundError:
            pass
        except NotADirectoryError:
            _remove_file(path)
        except Exception as e:
            logger.warning(f"Failed to remove tree {path}: {e}")

    def _remove_matching(pattern):
        for matched_path in glob.glob(pattern):
            _remove_file(matched_path)

    def _remove_matching_except(pattern, keep_paths):
        keep_paths = {os.path.abspath(path) for path in keep_paths if path is not None}
        for matched_path in glob.glob(pattern):
            if os.path.abspath(matched_path) not in keep_paths:
                _remove_file(matched_path)

    def _purge_staged_checkpoints():
        _remove_matching(os.path.join(checkpoint_stage_dir, "vjepa_camera_ckpt_*.pt"))

    def _prune_checkpoint_outputs(keep_path):
        _remove_matching_except(os.path.join(scratch_folder, "e*.pt"), {keep_path})
        _remove_matching_except(os.path.join(scratch_folder, "final.pt"), {keep_path})

    def _prune_training_curve_outputs(keep_path):
        _remove_matching_except(os.path.join(folder, "training_curves_e*.png"), {keep_path})

    def _prune_eval_summary_outputs(keep_path):
        _remove_matching_except(os.path.join(folder, "eval_summary_e*.json"), {keep_path})

    def _prune_pca_outputs(epoch):
        keep_pattern = os.path.join(pca_vis_dir, f"e{epoch:03d}_scene*.png")
        keep_paths = set(glob.glob(keep_pattern))
        if keep_paths:
            _remove_matching_except(os.path.join(pca_vis_dir, "e*_scene*.png"), keep_paths)

    def _free_bytes(path):
        target_path = path if os.path.exists(path) else (os.path.dirname(path) or ".")
        return shutil.disk_usage(target_path).free

    def _recursive_tensor_nbytes(obj):
        if torch.is_tensor(obj):
            return obj.numel() * obj.element_size()
        if isinstance(obj, dict):
            return sum(_recursive_tensor_nbytes(v) for v in obj.values())
        if isinstance(obj, (list, tuple)):
            return sum(_recursive_tensor_nbytes(v) for v in obj)
        return 0

    def _estimate_full_checkpoint_required_bytes():
        # Estimate serialized payload size from live model/optimizer tensors,
        # then add safety margin + free-space buffer for tmp-file staging.
        model_bytes = (
            _recursive_tensor_nbytes(encoder.state_dict())
            + _recursive_tensor_nbytes(predictor.state_dict())
            + _recursive_tensor_nbytes(target_encoder.state_dict())
        )
        optimizer_bytes = _recursive_tensor_nbytes(getattr(optimizer, "state", {}))
        scaler_state = None if scaler is None else scaler.state_dict()
        scaler_bytes = _recursive_tensor_nbytes(scaler_state)
        payload_bytes = model_bytes + optimizer_bytes + scaler_bytes
        return int(payload_bytes * FULL_CHECKPOINT_SIZE_SAFETY_FACTOR) + CHECKPOINT_SAVE_BUFFER_BYTES

    def _compact_state_dict(state_dict):
        compact_state = {}
        for key, value in state_dict.items():
            if torch.is_tensor(value):
                value = value.detach().cpu()
                if torch.is_floating_point(value):
                    value = value.to(torch.bfloat16)
            compact_state[key] = value
        return compact_state

    def _purge_runtime_caches():
        _purge_staged_checkpoints()
        pip_cache_dir = os.environ.get("PIP_CACHE_DIR")
        if pip_cache_dir:
            _remove_tree(pip_cache_dir)
            os.makedirs(pip_cache_dir, exist_ok=True)
        # Keep prewarmed pretrained checkpoints in TORCH_HOME between ablations.
        # Deleting them here causes later experiments in the same job to fail.

    def _unwrapped_state_dict(m):
        """Return state_dict of a possibly DDP-wrapped module WITHOUT the
        ``module.`` prefix. Checkpoints must be portable to single-GPU
        subprocesses (e.g. rollout eval) which build an unwrapped model."""
        return (m.module if hasattr(m, "module") else m).state_dict()

    def _build_full_checkpoint_payload(epoch):
        payload = {
            "encoder": _unwrapped_state_dict(encoder),
            "predictor": _unwrapped_state_dict(predictor),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": _unwrapped_state_dict(target_encoder),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "checkpoint_type": "full",
        }
        if corrector is not None:
            payload["corrector"] = _unwrapped_state_dict(corrector)
        if latent_patchgan is not None:
            payload["latent_patchgan"] = _unwrapped_state_dict(latent_patchgan)
        if latent_patchgan_optimizer is not None:
            payload["latent_patchgan_opt"] = latent_patchgan_optimizer.state_dict()
        return payload

    def _build_weights_only_payload(epoch, *, include_target_encoder):
        payload = {
            "encoder": _compact_state_dict(_unwrapped_state_dict(encoder)),
            "predictor": _compact_state_dict(_unwrapped_state_dict(predictor)),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "checkpoint_type": "weights_only_bf16",
        }
        if include_target_encoder:
            payload["target_encoder"] = _compact_state_dict(_unwrapped_state_dict(target_encoder))
        if corrector is not None:
            # Corrector weights are tiny (≤5M params); keep at fp32 for
            # numerical headroom on the residual scales.
            payload["corrector"] = _unwrapped_state_dict(corrector)
        if latent_patchgan is not None:
            payload["latent_patchgan"] = _compact_state_dict(_unwrapped_state_dict(latent_patchgan))
        return payload

    def _build_predictor_only_payload(epoch):
        payload = {
            "predictor": _compact_state_dict(_unwrapped_state_dict(predictor)),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "checkpoint_type": "weights_only_bf16_predictor",
        }
        if corrector is not None:
            payload["corrector"] = _unwrapped_state_dict(corrector)
        if latent_patchgan is not None:
            payload["latent_patchgan"] = _compact_state_dict(_unwrapped_state_dict(latent_patchgan))
        return payload

    def _write_checkpoint_payload(payload, path, *, tag):
        tmp_path = None
        try:
            _purge_staged_checkpoints()
            target_dir = os.path.dirname(path) or "."
            os.makedirs(target_dir, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=checkpoint_stage_dir,
                prefix="vjepa_camera_ckpt_",
                suffix=".pt",
                delete=False,
            ) as tf:
                tmp_path = tf.name
            torch.save(payload, tmp_path)
            same_filesystem = os.stat(tmp_path).st_dev == os.stat(target_dir).st_dev
            if same_filesystem:
                os.replace(tmp_path, path)
                tmp_path = None
            else:
                shutil.copy2(tmp_path, path)
            logger.info(f"Saved checkpoint ({tag}): {path}")
            return True
        except Exception as e:
            logger.warning(f"Failed checkpoint save ({tag}) to {path}: {e}")
            return False
        finally:
            if tmp_path is not None:
                _remove_file(tmp_path)

    start_epoch = 0
    if resume_path is not None and os.path.exists(resume_path):
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint(
            r_path=resume_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            corrector=corrector,
            latent_patchgan=latent_patchgan,
            latent_patchgan_opt=latent_patchgan_optimizer,
        )
        for _ in range(start_epoch * ipe):
            scheduler.step()
            wd_scheduler.step()

    _purge_runtime_caches()

    loss_meter = AverageMeter()

    def _broadcast_bool_from_rank0(flag):
        if not (dist.is_available() and dist.is_initialized() and world_size > 1):
            return bool(flag)
        flag_tensor = torch.tensor([1 if (rank == 0 and flag) else 0], device=device, dtype=torch.int32)
        dist.broadcast(flag_tensor, src=0)
        return bool(flag_tensor.item())

    def _safe_dist_barrier():
        if not (dist.is_available() and dist.is_initialized() and world_size > 1):
            return
        try:
            dist.barrier()
        except Exception as e:
            logger.warning(f"Distributed barrier failed: {e}")

    def _shutdown_distributed():
        if not (dist.is_available() and dist.is_initialized()):
            return
        if world_size > 1:
            try:
                dist.barrier()
            except Exception as e:
                logger.warning(f"Final distributed barrier failed: {e}")
        try:
            dist.destroy_process_group()
        except Exception as e:
            logger.warning(f"destroy_process_group failed: {e}")

    def save_checkpoint(epoch, path):
        if not checkpoint_saving_enabled:
            return False
        if rank != 0:
            return False
        _purge_runtime_caches()
        if checkpoint_save_mode == "weights_only_predictor":
            return _write_checkpoint_payload(
                _build_predictor_only_payload(epoch),
                path,
                tag="weights-only-bf16-predictor",
            )
        if checkpoint_save_mode == "weights_only_no_target":
            return _write_checkpoint_payload(
                _build_weights_only_payload(epoch, include_target_encoder=False),
                path,
                tag="weights-only-bf16-no-target",
            )
        if checkpoint_save_mode == "weights_only":
            if _write_checkpoint_payload(
                _build_weights_only_payload(epoch, include_target_encoder=True),
                path,
                tag="weights-only-bf16",
            ):
                return True
            return _write_checkpoint_payload(
                _build_weights_only_payload(epoch, include_target_encoder=False),
                path,
                tag="weights-only-bf16-no-target",
            )
        free_bytes = _free_bytes(checkpoint_stage_dir)
        full_required_bytes = max(
            MIN_FREE_BYTES_FOR_FULL_CHECKPOINT,
            _estimate_full_checkpoint_required_bytes(),
        )
        if free_bytes >= full_required_bytes:
            if _write_checkpoint_payload(_build_full_checkpoint_payload(epoch), path, tag="full"):
                return True
            logger.warning("Full checkpoint save failed; falling back to weights-only checkpoint.")
        else:
            if checkpoint_save_mode == "full":
                logger.warning(
                    f"Requested full checkpoint but only {free_bytes / 1024.0**3:.2f} GiB free in {checkpoint_stage_dir}; "
                    f"need about {full_required_bytes / 1024.0**3:.2f} GiB. Falling back to compact checkpoint."
                )
            logger.warning(
                f"Only {free_bytes / 1024.0**3:.2f} GiB free in {checkpoint_stage_dir}; "
                f"full checkpoint needs about {full_required_bytes / 1024.0**3:.2f} GiB. "
                "saving compact weights-only checkpoint instead of full checkpoint."
            )
        if _write_checkpoint_payload(
            _build_weights_only_payload(epoch, include_target_encoder=True),
            path,
            tag="weights-only-bf16",
        ):
            return True
        if _write_checkpoint_payload(
            _build_weights_only_payload(epoch, include_target_encoder=False),
            path,
            tag="weights-only-bf16-no-target",
        ):
            return True
        return False

    logger.info(
        "Initializing loader... num_workers=%s persistent_workers=%s pin_memory=%s",
        num_workers,
        bool(persistent_workers and num_workers > 0),
        pin_mem,
    )
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)
    logger.info("Loader iterator initialized.")

    if skip_batches > 0:
        logger.info(f"Skip {skip_batches} batches")
        for itr in range(skip_batches):
            if itr % 10 == 0:
                logger.info(f"Skip {itr}/{skip_batches} batches")
            try:
                _ = next(loader)
            except Exception:
                loader = iter(unsupervised_loader)
                _ = next(loader)

    if sync_gc:
        gc.disable()
        gc.collect()

    training_stats = {
        "loss": [],
        "loss_pred": [],
        "loss_ctx": [],
        "loss_step1": [],
        "loss_step2": [],
        "lr": [],
        "wd": [],
        "iter_ms": [],
        "gpu_ms": [],
        "mem_gb": [],
    }
    if corrector is not None:
        # Phase C metrics. Tracked per epoch alongside the standard losses
        # so plot_training_curves picks them up automatically (the visualizer
        # auto-plots unknown keys, see src/training/visualization.py).
        training_stats["loss_corr_step"] = []
        training_stats["loss_corr_denoise"] = []
        # Per-layer monitoring (TCR design §5.1).
        for _l in range(n_hierarchical_layers):
            training_stats[f"corr_input_norm_l{_l}"] = []
            training_stats[f"corr_output_norm_l{_l}"] = []
            training_stats[f"corr_delta_norm_l{_l}"] = []
            training_stats[f"corr_scale_l{_l}"] = []

    # Phase R metrics (TCR design §3.6). cycle_l1_h1 and cycle_weight are
    # only appended when ``cycle_enabled``; drift_from_prev is appended
    # whenever the meter has values (it is also useful as an identity-
    # collapse canary when the cycle loss is disabled, see TCR §3.6).
    if cycle_enabled:
        training_stats["cycle_l1_h1"] = []
        training_stats["cycle_weight"] = []
    # Phase R' metrics. composition_l1 is the primary gate signal; the
    # weight is logged separately so the warmup ramp is auditable from
    # the curve alone.
    if composition_enabled:
        training_stats["composition_l1"] = []
        training_stats["composition_weight"] = []
    if completion_enabled:
        training_stats["completion_l1"] = []
        training_stats["completion_visible_frac"] = []
        training_stats["completion_boundary_frac"] = []
        training_stats["completion_disoccluded_frac"] = []
    if latent_patchgan_enabled:
        training_stats["patchgan_g"] = []
        training_stats["patchgan_d"] = []
        training_stats["patchgan_weight"] = []
        training_stats["patchgan_reveal_frac"] = []
        training_stats["patchgan_pred_norm"] = []
        training_stats["patchgan_tgt_norm"] = []
        training_stats["patchgan_norm_ratio"] = []
    if target_slot_mode == "canvas_first":
        training_stats["canvas_valid_frac"] = []
        training_stats["canvas_oov_frac"] = []
        training_stats["canvas_boundary_frac"] = []
        training_stats["canvas_beta_valid"] = []
        training_stats["canvas_beta_boundary"] = []
        training_stats["canvas_beta_oov"] = []
        training_stats["canvas_input_norm_ratio"] = []
    training_stats["drift_from_prev"] = []

    def run_checkpoint_evaluation(epoch, previous_epoch=None):
        if rank != 0:
            return

        def _layerlist_to_h_eval(layer_outs):
            if normalize_reps:
                layer_outs = [
                    F.layer_norm(feat, (feat.size(-1),))
                    for feat in layer_outs
                ]
            return torch.cat(layer_outs, dim=-1)

        def _sample_latent_loss(preds, h_tgt):
            embed_dim = h_tgt.shape[-1] // n_hierarchical_layers
            pred_chunks = preds.split(embed_dim, dim=-1)
            h_chunks = h_tgt.split(embed_dim, dim=-1)
            loss_pred = preds.new_zeros(())
            for p, t in zip(pred_chunks, h_chunks):
                t = t.detach()
                if normalize_reps:
                    p = F.layer_norm(p, (p.size(-1),))
                    t = F.layer_norm(t, (t.size(-1),))
                if per_layer_balance:
                    # Rescale by target std so each layer's L1 contribution is
                    # dimensionless and equally weighted across hierarchical
                    # chunks. No-op when normalize_reps=True (std≈1 after LN).
                    scale = t.detach().std().clamp_min(1.0e-6)
                else:
                    scale = 1.0
                loss_pred = loss_pred + (
                    torch.mean(torch.abs(p - t) ** loss_exp) / loss_exp / scale
                )
            loss_pred = loss_pred / n_hierarchical_layers
            return float(loss_pred.item())

        def _sample_latent_loss_masked(preds, h_tgt, mask):
            mask = mask.to(device=preds.device, dtype=preds.dtype)
            if mask.ndim == 2:
                mask = mask.unsqueeze(-1)
            if float(mask.sum().detach().item()) <= 0.0:
                return None
            embed_dim = h_tgt.shape[-1] // n_hierarchical_layers
            pred_chunks = preds.split(embed_dim, dim=-1)
            h_chunks = h_tgt.split(embed_dim, dim=-1)
            loss_pred = preds.new_zeros(())
            token_norm = mask.sum().clamp_min(1.0)
            for p, t in zip(pred_chunks, h_chunks):
                t = t.detach()
                if normalize_reps:
                    p = F.layer_norm(p, (p.size(-1),))
                    t = F.layer_norm(t, (t.size(-1),))
                scale = t.detach().std().clamp_min(1.0e-6) if per_layer_balance else 1.0
                per_token = (torch.abs(p - t) ** loss_exp / loss_exp / scale).mean(dim=-1, keepdim=True)
                loss_pred = loss_pred + (per_token * mask).sum() / token_norm
            loss_pred = loss_pred / n_hierarchical_layers
            return float(loss_pred.item())

        def _rotation_degrees_from_quat(rel_quat):
            rel_quat = F.normalize(rel_quat, dim=-1, eps=1.0e-6)
            imag_norm = torch.linalg.norm(rel_quat[..., :3], dim=-1)
            real = rel_quat[..., 3].abs().clamp_min(1.0e-8)
            angle_rad = 2.0 * torch.atan2(imag_norm, real)
            return float(torch.rad2deg(angle_rad).item())

        def _summarize_motion_bins(metric_name, values, losses):
            if len(values) == 0:
                logger.warning(f"No motion-bin validation samples available for {metric_name}.")
                return None
            values_np = np.asarray(values, dtype=np.float32)
            losses_np = np.asarray(losses, dtype=np.float32)
            q1, q2 = np.quantile(values_np, [1.0 / 3.0, 2.0 / 3.0])
            bins = {
                "low": values_np <= q1,
                "medium": (values_np > q1) & (values_np <= q2),
                "high": values_np > q2,
            }
            parts = []
            summary = {
                "q33": float(q1),
                "q67": float(q2),
                "bins": {},
            }
            for label, mask in bins.items():
                count = int(mask.sum())
                if count > 0:
                    avg_loss = float(losses_np[mask].mean())
                    avg_value = float(values_np[mask].mean())
                    parts.append(f"{label}=loss {avg_loss:.5f} ({metric_name} {avg_value:.5f}, n={count})")
                    summary["bins"][label] = {
                        "loss": avg_loss,
                        "value": avg_value,
                        "count": count,
                    }
                else:
                    parts.append(f"{label}=loss nan ({metric_name} nan, n=0)")
                    summary["bins"][label] = {
                        "loss": None,
                        "value": None,
                        "count": 0,
                    }
            logger.info(
                f"Motion-bin eval [{metric_name}] q33={float(q1):.5f} q67={float(q2):.5f} :: "
                + " | ".join(parts)
            )
            return summary

        try:
            curve_path = os.path.join(folder, f"training_curves_e{epoch:03d}.png")
            plot_training_curves(
                training_stats,
                curve_path,
            )
            _prune_training_curve_outputs(curve_path)
            logger.info(f"Training curves saved to {curve_path}")
        except Exception as _ce:
            logger.warning(f"Training curves failed: {_ce}")

        if pca_loader is None:
            return

        logger.info("Running PCA feature visualisation...")
        grid_h = grid_w = crop_size // patch_size
        HW = grid_h * grid_w
        T_tok = seq_len // tubelet_size
        k_eval = max(1, min(n_ctx_tubelets, T_tok - 1))
        target_tubelet_idx = k_eval
        eval_limit = max(n_pca_scenes, n_motion_eval_scenes)
        _enc = target_encoder.module if hasattr(target_encoder, "module") else target_encoder
        _pred = predictor.module if hasattr(predictor, "module") else predictor
        _enc.eval()
        _pred.eval()
        pca_outputs_written = 0
        motion_eval_losses = []
        motion_eval_translation = []
        motion_eval_rotation_deg = []
        region_eval = {
            "visible": [],
            "boundary": [],
            "disoccluded": [],
        }
        eval_summary = {
            "epoch": int(epoch),
            "context_tubelets": int(k_eval),
            "target_tubelet_idx": int(target_tubelet_idx),
            "n_motion_eval_scenes": int(n_motion_eval_scenes),
            "n_fixed_stride_eval_scenes": int(n_fixed_stride_eval_scenes),
            "motion_bins": {},
            "fixed_stride": {},
            "artifacts": {
                "training_curves": f"training_curves_e{epoch:03d}.png",
                "pca_vis_dir": "pca_vis",
            },
        }

        def _forward_eval_sample(eval_sample, target_idx):
            v_imgs = eval_sample["images"].to(device)
            v_states = eval_sample["states"].to(device, dtype=torch.float)[:, ::tubelet_size]
            v_states = _canonicalize_states(v_states)
            v_actions = eval_sample["actions"].to(device, dtype=torch.float)[:, ::tubelet_size]
            v_intrinsics = eval_sample["intrinsics"].to(device, dtype=torch.float)[:, ::tubelet_size]
            full_clip = v_imgs.permute(0, 2, 1, 3, 4)
            # V-JEPA 2.1 encoder forward: returns a list of per-layer
            # (B, T_tok*HW, D) tensors where T_tok = T / tubelet_size.
            # With tubelet_size=1 (image path) this is one token grid per
            # frame; with tubelet_size=2 (legacy video path) adjacent frame
            # pairs are fused by patch_embed's Conv3d.
            layer_outs = _enc(full_clip)
            h_tgt_full = _layerlist_to_h_eval(layer_outs)
            h_tgt = h_tgt_full[0]
            layer_dim = layer_outs[-1].shape[-1]
            ctx_frames_eval = target_idx * tubelet_size
            target_start = target_idx * HW
            target_end = (target_idx + 1) * HW
            h_tgt_last = h_tgt[target_start:target_end, -layer_dim:]
            h_tgt_eval = h_tgt_full[:, target_start:target_end, :]
            ctx_clip = v_imgs[:, :ctx_frames_eval].permute(0, 2, 1, 3, 4)
            ctx_lo = _enc(ctx_clip)
            h_ctx = torch.cat(ctx_lo, dim=-1)
            B_v = h_ctx.shape[0]
            target_depth_eval = None
            if da3_warp_enabled:
                if completion_depth_teacher is None or compute_teacher_depths_batch_with_confidence is None:
                    raise RuntimeError("DA3 warp requested but depth teacher is not initialized.")
                depth_maps_warp_eval, _ = compute_teacher_depths_batch_with_confidence(
                    v_imgs[:, target_idx * tubelet_size].float(),
                    depth_processor=completion_depth_processor,
                    depth_model=completion_depth_teacher,
                    device=device,
                    H=v_imgs.shape[-2],
                    W=v_imgs.shape[-1],
                )
                target_depth_eval = depth_maps_warp_eval
            if target_slot_mode == "canvas_first" and hasattr(_pred, "build_target_canvas_inputs"):
                local_states_eval_canvas = _canonicalize_states(v_states[:, :target_idx + 1])
                local_actions_eval_canvas = v_actions[:, :target_idx + 1]
                local_intr_eval_canvas = (
                    v_intrinsics[:, :target_idx + 1] if use_intrinsics else None
                )
                canvas_inputs_v = _pred.build_target_canvas_inputs(
                    h_ctx,
                    local_states_eval_canvas,
                    local_actions_eval_canvas,
                    local_intr_eval_canvas,
                    target_depth=target_depth_eval,
                )
                boundary_v = _rotation_unseen_boundary_mask(
                    _pred,
                    local_states_eval_canvas,
                    local_intr_eval_canvas,
                    HW,
                    h_ctx.dtype,
                    intermediate_unseen_boundary_band,
                )
                eval_rot_weight = None
                if canvas_warp_mode == "rot_gated":
                    eval_rot_weight = _canvas_rot_weight(
                        local_actions_eval_canvas,
                        dtype=h_ctx.dtype,
                    ).to(device=device)
                future_tokens_v = _pred.prepare_canvas_slot(
                    warped_context_raw=canvas_inputs_v["warped_context_raw"],
                    valid_mask=canvas_inputs_v["valid_mask"],
                    oov_mask=canvas_inputs_v["oov_mask"],
                    boundary_mask=boundary_v,
                    confidence_mask=canvas_inputs_v["confidence_mask"],
                    rollout_step=max(0, target_idx - 1),
                    rot_weight=eval_rot_weight,
                )
            else:
                future_tokens_v = _pred.future_mask_token.to(device=device, dtype=h_ctx.dtype).expand(
                    B_v, HW, -1
                )
            h_in = torch.cat([h_ctx, future_tokens_v], dim=1)
            preds_v, _, _delta_v, _fsq_v, _warped_ctx_v = _pred(
                h_in,
                local_actions_eval_canvas if target_slot_mode == "canvas_first" else v_actions[:, :target_idx + 1],
                local_states_eval_canvas if target_slot_mode == "canvas_first" else v_states[:, :target_idx + 1],
                intrinsics=(
                    local_intr_eval_canvas
                    if target_slot_mode == "canvas_first"
                    else (v_intrinsics[:, :target_idx + 1] if use_intrinsics else None)
                ),
                target_depth=target_depth_eval,
            )
            preds_v_next = _reconstruct_residual_prediction(
                preds_v[:, -HW:, :],
                _warped_ctx_v,
            )
            region_masks = None
            if completion_enabled:
                local_states_eval = v_states[:, target_idx - 1:target_idx + 1]
                local_intr_eval = (
                    v_intrinsics[:, target_idx - 1:target_idx + 1]
                    if use_intrinsics else None
                )
                if (
                    completion_mask_source == "da3"
                    and completion_depth_teacher is not None
                    and compute_teacher_depths_batch_with_confidence is not None
                ):
                    depth_imgs_eval = torch.cat(
                        [
                            v_imgs[:, (target_idx - 1) * tubelet_size],
                            v_imgs[:, target_idx * tubelet_size],
                        ],
                        dim=0,
                    )
                    depth_maps_eval, _ = compute_teacher_depths_batch_with_confidence(
                        depth_imgs_eval.float(),
                        depth_processor=completion_depth_processor,
                        depth_model=completion_depth_teacher,
                        device=device,
                        H=v_imgs.shape[-2],
                        W=v_imgs.shape[-1],
                    )
                    region_masks = _depth_visibility_region_masks(
                        _pred,
                        local_states_eval,
                        local_intr_eval,
                        depth_maps_eval[:1],
                        depth_maps_eval[1:],
                        HW,
                        preds_v.dtype,
                        completion_depth_tau,
                        completion_boundary_dilate,
                    )
                else:
                    boundary = _rotation_unseen_boundary_mask(
                        _pred,
                        local_states_eval,
                        local_intr_eval,
                        HW,
                        preds_v.dtype,
                        intermediate_unseen_boundary_band,
                    )
                    if boundary is not None:
                        boundary = boundary.to(device=preds_v.device, dtype=preds_v.dtype).clamp(0.0, 1.0)
                        region_masks = {
                            "visible": 1.0 - boundary,
                            "boundary": boundary,
                            "disoccluded": torch.zeros_like(boundary),
                        }
            return {
                "imgs": v_imgs,
                "preds_next": preds_v_next,
                "h_tgt_eval": h_tgt_eval,
                "h_tgt_last": h_tgt_last,
                "h_pred_last": preds_v_next[0, :, -layer_dim:],
                "rel_action": v_actions[0, target_idx - 1],
                "region_masks": region_masks,
            }

        def _build_pca_vis_sample(eval_sample):
            pca_target_indices = list(range(1, T_tok))
            pca_targets_gt = []
            pca_targets_pred = []
            pca_rgb_frames = []
            pca_frame_labels = []
            for pca_target_idx in pca_target_indices:
                sample_out = _forward_eval_sample(eval_sample, pca_target_idx)
                pca_targets_gt.append(sample_out["h_tgt_last"])
                pca_targets_pred.append(sample_out["h_pred_last"])
                pca_rgb_frames.append(sample_out["imgs"][0, pca_target_idx * tubelet_size])
                pca_frame_labels.append(f"tubelet {pca_target_idx}")
            return {
                "h_gt": torch.cat(pca_targets_gt, dim=0),
                "h_pred": torch.cat(pca_targets_pred, dim=0),
                "imgs": torch.stack(pca_rgb_frames, dim=0),
                "frame_labels": pca_frame_labels,
            }

        def _run_fixed_stride_eval(eval_loader, max_scenes):
            losses = []
            with torch.no_grad():
                for eval_idx, eval_sample in enumerate(eval_loader):
                    if eval_idx >= max_scenes:
                        break
                    sample_out = _forward_eval_sample(eval_sample, target_tubelet_idx)
                    losses.append(_sample_latent_loss(sample_out["preds_next"], sample_out["h_tgt_eval"]))
            return losses

        try:
            with torch.no_grad():
                for vis_idx, vis_sample in enumerate(pca_loader):
                    if vis_idx >= eval_limit:
                        break
                    sample_out = _forward_eval_sample(vis_sample, target_tubelet_idx)
                    if vis_idx < n_motion_eval_scenes:
                        motion_eval_losses.append(_sample_latent_loss(sample_out["preds_next"], sample_out["h_tgt_eval"]))
                        if sample_out.get("region_masks") is not None:
                            for _region_name, _region_mask in sample_out["region_masks"].items():
                                _region_loss = _sample_latent_loss_masked(
                                    sample_out["preds_next"],
                                    sample_out["h_tgt_eval"],
                                    _region_mask,
                                )
                                if _region_loss is not None:
                                    region_eval[_region_name].append(_region_loss)
                        rel_action = sample_out["rel_action"]
                        motion_eval_translation.append(float(torch.linalg.norm(rel_action[:3]).item()))
                        motion_eval_rotation_deg.append(_rotation_degrees_from_quat(rel_action[3:]))
                    if vis_idx < n_pca_scenes:
                        out_path = os.path.join(pca_vis_dir, f"e{epoch:03d}_scene{vis_idx:02d}.png")
                        try:
                            pca_sample = _build_pca_vis_sample(vis_sample)
                            visualize_pca_features(
                                h_gt=pca_sample["h_gt"],
                                h_pred=pca_sample["h_pred"],
                                imgs=pca_sample["imgs"],
                                grid_h=grid_h,
                                grid_w=grid_w,
                                out_path=out_path,
                                n_frames=len(pca_sample["frame_labels"]),
                                frame_labels=pca_sample["frame_labels"],
                            )
                            pca_outputs_written += 1
                        except Exception as _ve:
                            logger.warning(f"PCA vis failed for scene {vis_idx}: {_ve}")
        finally:
            _enc.train()
            _pred.train()

        translation_summary = _summarize_motion_bins("translation", motion_eval_translation, motion_eval_losses)
        rotation_summary = _summarize_motion_bins("rotation_deg", motion_eval_rotation_deg, motion_eval_losses)
        if translation_summary is not None:
            eval_summary["motion_bins"]["translation"] = translation_summary
        if rotation_summary is not None:
            eval_summary["motion_bins"]["rotation_deg"] = rotation_summary
        if completion_enabled:
            eval_summary["region_one_step"] = {
                name: {
                    "loss": (float(np.mean(values)) if values else None),
                    "count": int(len(values)),
                }
                for name, values in region_eval.items()
            }
            logger.info(
                "Region one-step eval visible/boundary/disoccluded = %s / %s / %s",
                eval_summary["region_one_step"]["visible"]["loss"],
                eval_summary["region_one_step"]["boundary"]["loss"],
                eval_summary["region_one_step"]["disoccluded"]["loss"],
            )
        for stride_label, (stride_value, stride_loader) in fixed_stride_eval_loaders.items():
            stride_losses = _run_fixed_stride_eval(stride_loader, n_fixed_stride_eval_scenes)
            if len(stride_losses) == 0:
                logger.warning(f"No fixed-stride eval samples available for {stride_label} stride={stride_value}.")
                eval_summary["fixed_stride"][stride_label] = {
                    "stride": int(stride_value),
                    "loss": None,
                    "count": 0,
                }
            else:
                mean_stride_loss = float(np.mean(stride_losses))
                logger.info(
                    f"Fixed-stride eval [{stride_label}] stride={stride_value} loss={mean_stride_loss:.5f} n={len(stride_losses)}"
                )
                eval_summary["fixed_stride"][stride_label] = {
                    "stride": int(stride_value),
                    "loss": mean_stride_loss,
                    "count": int(len(stride_losses)),
                }

        eval_summary_path = os.path.join(folder, f"eval_summary_e{epoch:03d}.json")
        try:
            with open(eval_summary_path, "w") as f:
                json.dump(eval_summary, f, separators=(",", ":"), sort_keys=True)
                f.write("\n")
            _prune_eval_summary_outputs(eval_summary_path)
            logger.info(f"Eval summary saved to {eval_summary_path}")
        except Exception as e:
            logger.warning(f"Failed to write eval summary to {eval_summary_path}: {e}")

        # ------------------------------------------------------------------
        # Mirror eval scalars into training_stats so plot_training_curves
        # renders them alongside train losses. Unknown keys are auto-plotted
        # by src/training/visualization.py (future-proofing branch).
        # ------------------------------------------------------------------
        def _bin_loss(bin_summary, bin_name):
            if bin_summary is None:
                return None
            bins = bin_summary.get("bins", {}) if isinstance(bin_summary, dict) else {}
            entry = bins.get(bin_name)
            if not isinstance(entry, dict):
                return None
            val = entry.get("loss")
            return float(val) if isinstance(val, (int, float)) else None

        if motion_eval_losses:
            training_stats.setdefault("eval_motion_mean_loss", []).append(
                float(np.mean(motion_eval_losses))
            )
        for bin_name in ("low", "medium", "high"):
            t_val = _bin_loss(translation_summary, bin_name)
            r_val = _bin_loss(rotation_summary, bin_name)
            if t_val is not None:
                training_stats.setdefault(f"eval_motion_trans_{bin_name}", []).append(t_val)
            if r_val is not None:
                training_stats.setdefault(f"eval_motion_rot_{bin_name}", []).append(r_val)
        for stride_label, stride_entry in eval_summary.get("fixed_stride", {}).items():
            if not isinstance(stride_entry, dict):
                continue
            loss_val = stride_entry.get("loss")
            if isinstance(loss_val, (int, float)):
                training_stats.setdefault(f"eval_fixed_stride_{stride_label}", []).append(float(loss_val))

        # Re-plot the curves now that the eval series have been updated, so
        # the current epoch's PNG actually reflects this epoch's eval point.
        try:
            plot_training_curves(training_stats, curve_path)
        except Exception as _ce:
            logger.warning(f"Training curves refresh (with eval) failed: {_ce}")

        if pca_outputs_written > 0:
            _prune_pca_outputs(epoch)
            logger.info(f"PCA visualisations saved to {pca_vis_dir}")
        else:
            logger.warning("No PCA visualisations were written; keeping previous PCA outputs.")

        # ------------------------------------------------------------------ #
        # In-process camera rollout eval (replaces the 3× cold-start
        # ``run_rollout_evals`` loop in scripts/train_camera_re10k.sh).
        # Enabled by setting CAMERA_ROLLOUT_EVAL_HORIZONS (e.g. "5,12,20").
        # One rollout at max(horizons) yields per-horizon summaries as
        # prefixes — so 3 legacy horizons cost one forward pass with the
        # already-loaded encoder/predictor (zero weight reload).
        # ------------------------------------------------------------------ #
        _horizons_env = os.environ.get("CAMERA_ROLLOUT_EVAL_HORIZONS", "").strip()
        if _horizons_env:
            try:
                horizons_list = [int(h) for h in _horizons_env.split(",") if h.strip()]
            except ValueError as _he:
                logger.warning(f"Ignoring CAMERA_ROLLOUT_EVAL_HORIZONS={_horizons_env!r}: {_he}")
                horizons_list = []
            if horizons_list:
                try:
                    # Closure-captured from the outer function; NameError if the
                    # eval loader init block above failed (then we skip).
                    _lazy_ref = _eval_lazy  # noqa: F821 -- captured via closure
                    _gwm_root_ref = gwm_root  # noqa: F821 -- captured via closure
                    tools_dir = os.path.join(_gwm_root_ref, "tools")
                    if tools_dir not in sys.path:
                        sys.path.insert(0, tools_dir)
                    from camera_rollout_pca import (
                        build_rollout_datasets_from_lazy,
                        run_rollout_eval_inprocess,
                    )

                    max_horizon = max(horizons_list)
                    # Seed context for the rollout = max_ctx_tubelets (same as
                    # the tool's default); guard against running off the end.
                    seed_tubelets_eval = max(1, min(max_ctx_tubelets, total_tubelets - 1))
                    context_tubelets_eval = max(1, min(n_ctx_tubelets, seed_tubelets_eval))
                    rollout_seq_len = tubelet_size * (seed_tubelets_eval + max_horizon)
                    one_step_seq_len = tubelet_size * (context_tubelets_eval + 1)

                    rollout_ds = build_rollout_datasets_from_lazy(
                        lazy_dataset=_lazy_ref, seq_len=rollout_seq_len,
                        stride=stride, min_stride=min_stride, stride_values=stride_values,
                        image_size=crop_size,
                    )
                    one_step_ds = build_rollout_datasets_from_lazy(
                        lazy_dataset=_lazy_ref, seq_len=one_step_seq_len,
                        stride=stride, min_stride=min_stride, stride_values=stride_values,
                        image_size=crop_size,
                    )
                    fixed_stride_datasets = {
                        label: build_rollout_datasets_from_lazy(
                            lazy_dataset=_lazy_ref, seq_len=one_step_seq_len,
                            stride=sv, min_stride=sv, stride_values=[sv],
                            image_size=crop_size,
                        )
                        for label, sv in [
                            ("small", eval_stride_small),
                            ("medium", eval_stride_medium),
                            ("large", eval_stride_large),
                        ]
                    }

                    rollout_out_dir = Path(folder) / f"rollout_eval_e{epoch:03d}"
                    _enc_unwrapped = encoder.module if hasattr(encoder, "module") else encoder
                    _pred_unwrapped = predictor.module if hasattr(predictor, "module") else predictor
                    _tgt_unwrapped = target_encoder.module if hasattr(target_encoder, "module") else target_encoder
                    dtype_name_eval = (
                        "bfloat16"
                        if getattr(torch, "get_autocast_gpu_dtype", lambda: torch.bfloat16)() is torch.bfloat16
                        else "float16"
                    )

                    rollout_summary = run_rollout_eval_inprocess(
                        encoder=_enc_unwrapped,
                        predictor=_pred_unwrapped,
                        target_encoder=_tgt_unwrapped,
                        rollout_ds=rollout_ds,
                        one_step_ds=one_step_ds,
                        fixed_stride_datasets=fixed_stride_datasets,
                        device=device,
                        output_dir=rollout_out_dir,
                        horizons=horizons_list,
                        n_scenes=max(1, n_pca_scenes),
                        scene_start_index=0,
                        context_tubelets=context_tubelets_eval,
                        seed_tubelets=seed_tubelets_eval,
                        tubelet_size=tubelet_size,
                        crop_size=crop_size,
                        patch_size=patch_size,
                        n_hierarchical_layers=n_hierarchical_layers,
                        normalize_reps=normalize_reps,
                        loss_exp=loss_exp,
                        use_intrinsics=use_intrinsics,
                        n_motion_eval_scenes=n_motion_eval_scenes,
                        n_fixed_stride_eval_scenes=n_fixed_stride_eval_scenes,
                        eval_stride_small=eval_stride_small,
                        eval_stride_medium=eval_stride_medium,
                        eval_stride_large=eval_stride_large,
                        autocast_enabled=True,
                        dtype_name=dtype_name_eval,
                        per_layer_balance=per_layer_balance,
                        residual_target=(residual_target and residual_target_reconstruct),
                        eval_seed=int(os.environ.get("CAMERA_ROLLOUT_EVAL_SEED", "0")),
                    )
                    logger.info(
                        "In-process camera rollout eval horizons=%s -> %s",
                        horizons_list, rollout_out_dir,
                    )
                    # Expose per-horizon aggregate to the curve plotter.
                    for key, hsum in rollout_summary.items():
                        open_vec = hsum.get("open_loop_loss_by_step_mean") or []
                        closed_vec = hsum.get("closed_loop_loss_by_step_mean") or []
                        if open_vec:
                            training_stats.setdefault(
                                f"eval_rollout_{key}_open_mean", []
                            ).append(float(np.mean(open_vec)))
                        if closed_vec:
                            training_stats.setdefault(
                                f"eval_rollout_{key}_closed_mean", []
                            ).append(float(np.mean(closed_vec)))
                except Exception as _re:
                    logger.warning(f"In-process rollout eval failed: {_re}")

    # ------------------------------------------------------------------ #
    # TRAINING LOOP
    # ------------------------------------------------------------------ #
    for epoch in range(start_epoch, num_epochs):
        logger.info("Epoch %d" % (epoch + 1))
        unsupervised_sampler.set_epoch(epoch)
        loader = iter(unsupervised_loader)

        loss_meter = AverageMeter()
        loss_pred_meter = AverageMeter()
        loss_ctx_meter = AverageMeter()
        loss_step1_meter = AverageMeter()
        loss_step2_meter = AverageMeter()
        loss_corr_step_meter = AverageMeter()
        loss_corr_denoise_meter = AverageMeter()
        loss_completion_meter = AverageMeter()
        patchgan_g_meter = AverageMeter()
        patchgan_d_meter = AverageMeter()
        patchgan_reveal_frac_meter = AverageMeter()
        patchgan_pred_norm_meter = AverageMeter()
        patchgan_tgt_norm_meter = AverageMeter()
        patchgan_norm_ratio_meter = AverageMeter()
        completion_visible_meter = AverageMeter()
        completion_boundary_meter = AverageMeter()
        completion_disoccluded_meter = AverageMeter()
        birth_delta_norm_meter = AverageMeter()
        birth_action_norm_meter = AverageMeter()
        birth_mask_norm_meter = AverageMeter()
        birth_M_mean_meter = AverageMeter()
        birth_oov_mean_meter = AverageMeter()
        birth_bnd_mean_meter = AverageMeter()
        action_in_norm_meter = AverageMeter()
        action_out_norm_meter = AverageMeter()
        action_trans_cos_meter = AverageMeter()
        action_trans_delta_meter = AverageMeter()
        action_trans_ratio_meter = AverageMeter()
        action_rot_delta_meter = AverageMeter()
        birth_raw_err_meter = AverageMeter()
        birth_comp_err_meter = AverageMeter()
        birth_final_err_meter = AverageMeter()
        birth_raw_layer_meters = [AverageMeter() for _ in range(n_hierarchical_layers)]
        birth_comp_layer_meters = [AverageMeter() for _ in range(n_hierarchical_layers)]
        birth_final_layer_meters = [AverageMeter() for _ in range(n_hierarchical_layers)]
        # P0.5/P0.6: per-layer prediction L1 and warp-depth diagnostics.
        # layer_loss_L{i} = mean abs error in the i-th hierarchical layer.
        # depth_std = spatial std of the depth-probe target depth (proxy for
        #   scene complexity / parallax difficulty in the current batch).
        # warp_valid/boundary/reveal = region mask fractions from completion.
        depth_std_meter = AverageMeter()
        depth_cv_meter = AverageMeter()
        warp_valid_meter = AverageMeter()
        warp_boundary_meter = AverageMeter()
        warp_reveal_meter = AverageMeter()
        canvas_valid_meter = AverageMeter()
        canvas_oov_meter = AverageMeter()
        canvas_boundary_meter = AverageMeter()
        canvas_beta_valid_meter = AverageMeter()
        canvas_beta_boundary_meter = AverageMeter()
        canvas_beta_oov_meter = AverageMeter()
        canvas_input_norm_ratio_meter = AverageMeter()
        canvas_warp_norm_meter = AverageMeter()
        canvas_mask_token_norm_meter = AverageMeter()
        canvas_mask_embed_norm_meter = AverageMeter()
        canvas_type_embed_norm_meter = AverageMeter()
        canvas_block_mask_frac_meter = AverageMeter()
        layer_loss_meters = [AverageMeter() for _ in range(4)]
        # Phase R diagnostics (TCR design section 3.6). Cycle loss is the
        # primary reversibility signal; drift_from_prev is the identity-
        # collapse early-warning indicator.
        loss_cycle_meter = AverageMeter()
        # Phase R' diagnostics. composition_l1 is the supervised-anchor
        # latent L1 between the composed-action prediction and the GT
        # latent at t+2; the path-path consistency term (when enabled)
        # is folded in here as well.
        loss_composition_meter = AverageMeter()
        drift_from_prev_meter = AverageMeter()
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()
        # Snapshot of the most recent corrector per-layer stats for end-of-
        # epoch logging. Updated only when log_freq fires (see hook above).
        last_corrector_layer_stats = None

        # Phase R: ramped cycle weight is a function of ``epoch`` only, so
        # compute it once per epoch instead of per-iter. This also keeps
        # ``_cycle_weight_now`` defined when ``log_stats`` runs even if
        # zero iters completed (e.g. epoch ended before any train step).
        if cycle_enabled:
            if epoch < cycle_start_epoch:
                _cycle_weight_now = 0.0
            elif epoch >= cycle_ramp_end_epoch:
                _cycle_weight_now = cycle_weight
            else:
                _cy_ramp = (
                    float(epoch - cycle_start_epoch)
                    / float(max(1, cycle_ramp_end_epoch - cycle_start_epoch))
                )
                _cycle_weight_now = cycle_weight * _cy_ramp
        else:
            _cycle_weight_now = 0.0

        # Phase R': same ramped-weight protocol as the cycle loss. Independent
        # schedule so R + R' can ramp at different speeds when stacked.
        if composition_enabled:
            if epoch < composition_start_epoch:
                _composition_weight_now = 0.0
            elif epoch >= composition_ramp_end_epoch:
                _composition_weight_now = composition_weight
            else:
                _co_ramp = (
                    float(epoch - composition_start_epoch)
                    / float(max(1, composition_ramp_end_epoch - composition_start_epoch))
                )
                _composition_weight_now = composition_weight * _co_ramp
        else:
            _composition_weight_now = 0.0

        # Phase E1: rotation-homography correspondence bias ramp. Same
        # epoch-only ramp protocol as Phase R / R'. The trainer writes the
        # scalar onto ``predictor.correspondence_bias_ramp`` (a non-persistent
        # buffer) so the predictor's forward picks it up next call.
        if correspondence_bias_enabled:
            if epoch < correspondence_bias_start_epoch:
                _correspondence_bias_ramp_now = 0.0
            elif epoch >= correspondence_bias_ramp_end_epoch:
                _correspondence_bias_ramp_now = 1.0
            else:
                _cb_ramp = (
                    float(epoch - correspondence_bias_start_epoch)
                    / float(max(1, correspondence_bias_ramp_end_epoch - correspondence_bias_start_epoch))
                )
                _correspondence_bias_ramp_now = _cb_ramp
            # Update the predictor's ramp buffer (handles DDP / compile-wrapped
            # modules by walking through the conventional .module attribute).
            _pred_for_ramp = predictor.module if hasattr(predictor, "module") else predictor
            if hasattr(_pred_for_ramp, "correspondence_bias_ramp"):
                _pred_for_ramp.correspondence_bias_ramp.fill_(_correspondence_bias_ramp_now)
        else:
            _correspondence_bias_ramp_now = 0.0

        if latent_patchgan_enabled:
            if epoch < latent_patchgan_start_epoch:
                _latent_patchgan_weight_now = 0.0
            elif latent_patchgan_ramp_epochs > 0:
                # Epochs in the loop are zero-based, while logs/checkpoints are
                # human-facing one-based. Treat ``start_epoch`` as the first
                # active epoch so start=0 gives a small nonzero weight in
                # logged Epoch 1 instead of waiting until Epoch 2.
                _pg_ramp = (
                    float(epoch - latent_patchgan_start_epoch + 1)
                    / float(max(1, latent_patchgan_ramp_epochs))
                )
                _latent_patchgan_weight_now = latent_patchgan_weight * min(1.0, max(0.0, _pg_ramp))
            else:
                _latent_patchgan_weight_now = latent_patchgan_weight
            if latent_patchgan is not None:
                if latent_patchgan_train_discriminator:
                    latent_patchgan.train()
                else:
                    latent_patchgan.eval()
        else:
            _latent_patchgan_weight_now = 0.0

        predictor_ref = predictor.module if hasattr(predictor, "module") else predictor

        for itr in range(ipe):
            itr_start_time = time.time()

            iter_retries = 0
            iter_successful = False
            while not iter_successful:
                try:
                    sample = next(loader)
                    iter_successful = True
                except StopIteration:
                    logger.info("Exhausted data loaders. Refreshing...")
                    unsupervised_sampler.set_epoch(epoch)
                    loader = iter(unsupervised_loader)
                except Exception as e:
                    NUM_RETRIES = 5
                    if iter_retries < NUM_RETRIES:
                        logger.warning(f"Data load exception (retry {iter_retries}): {e}")
                        iter_retries += 1
                        time.sleep(5)
                    else:
                        raise e

            def load_batch():
                imgs = sample["images"].to(device, non_blocking=True)
                s = sample["states"].to(device, dtype=torch.float, non_blocking=True)
                a = sample["actions"].to(device, dtype=torch.float, non_blocking=True)
                k = sample["intrinsics"].to(device, dtype=torch.float, non_blocking=True)
                states = s[:, ::tubelet_size, :]
                actions = a[:, ::tubelet_size, :]
                intrinsics = k[:, ::tubelet_size, :]
                return imgs, states, actions, intrinsics

            imgs, states, actions, intrinsics = load_batch()
            B, T_seq, C, H, W = imgs.shape
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                HW = (crop_size // patch_size) ** 2

                def _layerlist_to_h(layer_outs):
                    if normalize_reps:
                        layer_outs = [
                            F.layer_norm(feat, (feat.size(-1),))
                            for feat in layer_outs
                        ]
                    return torch.cat(layer_outs, dim=-1)

                with torch.no_grad():
                    full_clip = imgs.permute(0, 2, 1, 3, 4)
                    # target_encoder returns per-layer (B, T_tok*HW, D) tensors
                    # where T_tok = T / tubelet_size; concat over hierarchical
                    # layers gives (B, T_tok*HW, sum_D). Works for both the
                    # image path (tubelet_size=1) and the legacy video path
                    # (tubelet_size=2).
                    _enc_dtype = torch.bfloat16 if mixed_precision else None
                    _enc_ctx = (
                        torch.amp.autocast("cuda", dtype=_enc_dtype)
                        if _enc_dtype is not None
                        else contextlib.nullcontext()
                    )
                    with _enc_ctx:
                        h_target = _layerlist_to_h(target_encoder(full_clip))
                    h_target_tubelets = [
                        h_target[:, tubelet_idx * HW:(tubelet_idx + 1) * HW, :]
                        for tubelet_idx in range(total_tubelets)
                    ]

                if ar_random_context:
                    sampled_k_ctx = random.randint(min_ctx_tubelets, max_ctx_tubelets)
                else:
                    sampled_k_ctx = n_ctx_tubelets
                effective_rollout_steps = min(rollout_train_steps, total_tubelets - 1)
                k_ctx = max(1, min(int(sampled_k_ctx), total_tubelets - effective_rollout_steps))
                max_local_start = total_tubelets - k_ctx - effective_rollout_steps
                if ar_random_local_window and max_local_start > 0:
                    local_start = random.randint(0, max_local_start)
                else:
                    local_start = 0

                ctx_frame_start = local_start * tubelet_size
                ctx_frame_end = (local_start + k_ctx) * tubelet_size
                ctx_clip = imgs[:, ctx_frame_start:ctx_frame_end, :, :, :].permute(0, 2, 1, 3, 4)
                with torch.no_grad():
                    _enc_dtype = torch.bfloat16 if mixed_precision else None
                    _enc_ctx = (
                        torch.amp.autocast("cuda", dtype=_enc_dtype)
                        if _enc_dtype is not None
                        else contextlib.nullcontext()
                    )
                    with _enc_ctx:
                        ctx_layer_outs = encoder(ctx_clip)
                h_context = torch.cat(ctx_layer_outs, dim=-1)
                context_latents = [
                    h_context[:, tubelet_idx * HW:(tubelet_idx + 1) * HW, :]
                    for tubelet_idx in range(k_ctx)
                ]
                predictor_ref = predictor.module if hasattr(predictor, "module") else predictor
                completion_depths = None
                da3_depths = None
                if (completion_enabled and completion_mask_source == "da3") or da3_warp_enabled or canvas_da3_boundary_enabled or latent_patchgan_da3_enabled:
                    if completion_depth_teacher is None or compute_teacher_depths_batch_with_confidence is None:
                        raise RuntimeError("DA3 depth requested but depth teacher is not initialized.")
                    with torch.no_grad():
                        depth_imgs = imgs[:, ::tubelet_size].reshape(
                            B * total_tubelets, C, H, W
                        )
                        if completion_da3_depth_res > 0 and (
                            H != completion_da3_depth_res or W != completion_da3_depth_res
                        ):
                            depth_imgs_in = F.interpolate(
                                depth_imgs.float(),
                                size=(completion_da3_depth_res, completion_da3_depth_res),
                                mode="bilinear",
                                align_corners=False,
                            )
                        else:
                            depth_imgs_in = depth_imgs.float()
                        depth_maps, _depth_conf = compute_teacher_depths_batch_with_confidence(
                            depth_imgs_in,
                            depth_processor=completion_depth_processor,
                            depth_model=completion_depth_teacher,
                            device=device,
                            H=H,
                            W=W,
                        )
                        da3_depths = depth_maps.reshape(B, total_tubelets, H, W)
                        if (completion_enabled and completion_mask_source == "da3") or latent_patchgan_da3_enabled:
                            completion_depths = da3_depths

                def forward_predictor_with_trajectory(
                    step_context_latents,
                    local_states_canon,
                    local_actions_window,
                    local_intrinsics_window,
                    rollout_step=0,
                    target_depth=None,
                ):
                    """Predictor forward pass with explicit, pre-canonicalized
                    trajectory tensors.

                    Used by:
                      - the standard forward rollout (via ``forward_predictor``,
                        a thin wrapper that slices the outer-scope trajectory
                        tensors), and
                      - the Phase R reversibility cycle loss, which builds a
                        reversed/backward trajectory and reuses this helper.

                    Args:
                        step_context_latents: list of ``(B, HW, D_concat)`` context
                            latents in slot order.
                        local_states_canon: ``(B, len(step_context_latents) + 1, 7)``
                            states already canonicalized to the first slot.
                        local_actions_window: ``(B, len(step_context_latents) + 1, 7)``
                            relative SE(3) actions; the last entry is unused for
                            prediction (no next-frame to move to).
                        local_intrinsics_window: ``(B, len(step_context_latents) + 1, 4)``
                            or ``None`` when ``use_intrinsics`` is False.
                    """
                    h_context_step = torch.cat(step_context_latents, dim=1)
                    B_sz = h_context_step.shape[0]
                    if target_slot_mode == "canvas_first" and hasattr(predictor_ref, "build_target_canvas_inputs"):
                        canvas_inputs = predictor_ref.build_target_canvas_inputs(
                            h_context_step,
                            local_states_canon,
                            local_actions_window,
                            local_intrinsics_window,
                            target_depth=target_depth,
                        )
                        # LM7: DA3 depth-based boundary as canvas side-conditioning.
                        # Falls back to rotation-homography when da3_depths is None.
                        if canvas_da3_boundary_enabled and da3_depths is not None:
                            src_tubelet = max(0, rollout_step - 1) if rollout_step > 0 else 0
                            tgt_tubelet = min(rollout_step, da3_depths.shape[1] - 1)
                            da3_masks = _depth_visibility_region_masks(
                                predictor_ref,
                                local_states_canon,
                                local_intrinsics_window,
                                da3_depths[:, src_tubelet],
                                da3_depths[:, tgt_tubelet],
                                HW,
                                h_context_step.dtype,
                                completion_depth_tau,
                                completion_boundary_dilate,
                            )
                            if da3_masks is not None:
                                boundary_canvas = da3_masks["boundary"].reshape(B_sz, HW, 1)
                            else:
                                boundary_canvas = _rotation_unseen_boundary_mask(
                                    predictor_ref,
                                    local_states_canon,
                                    local_intrinsics_window,
                                    HW,
                                    h_context_step.dtype,
                                    intermediate_unseen_boundary_band,
                                )
                        else:
                            boundary_canvas = _rotation_unseen_boundary_mask(
                                predictor_ref,
                                local_states_canon,
                                local_intrinsics_window,
                                HW,
                                h_context_step.dtype,
                                intermediate_unseen_boundary_band,
                            )
                        # LM5c: rotation-gated beta — scale beta_valid per sample
                        # by how rotation-dominant the incoming action is.
                        canvas_rot_weight = None
                        if canvas_warp_mode == "rot_gated":
                            canvas_rot_weight = _canvas_rot_weight(
                                local_actions_window,
                                dtype=h_context_step.dtype,
                            ).to(device=h_context_step.device)
                        future_tokens = predictor_ref.prepare_canvas_slot(
                            warped_context_raw=canvas_inputs["warped_context_raw"],
                            valid_mask=canvas_inputs["valid_mask"],
                            oov_mask=canvas_inputs["oov_mask"],
                            boundary_mask=boundary_canvas,
                            confidence_mask=canvas_inputs["confidence_mask"],
                            rollout_step=rollout_step,
                            rot_weight=canvas_rot_weight,
                        )
                    else:
                        future_tokens = predictor_ref.future_mask_token.to(
                            device=h_context_step.device,
                            dtype=h_context_step.dtype,
                        )
                        future_tokens = future_tokens.expand(B_sz, HW, -1)
                    h_predictor_input = torch.cat([h_context_step, future_tokens], dim=1)
                    preds, _, delta_preds, fsq_logits, warped_ctx_raw = predictor(
                        h_predictor_input,
                        local_actions_window,
                        local_states_canon,
                        intrinsics=local_intrinsics_window,
                        target_depth=target_depth,
                    )
                    pred_slice = _reconstruct_residual_prediction(
                        preds[:, -HW:, :],
                        warped_ctx_raw,
                    )
                    fsq_slice = fsq_logits[:, -HW:, :, :] if fsq_logits is not None else None
                    if delta_preds is not None:
                        return pred_slice, delta_preds[:, -HW:, :], fsq_slice, warped_ctx_raw
                    return pred_slice, None, fsq_slice, warped_ctx_raw

                def _get_local_trajectory(step_context_latents, target_tubelet_idx):
                    current_local_start = target_tubelet_idx - len(step_context_latents)
                    local_states = _canonicalize_states(
                        states[:, current_local_start:target_tubelet_idx + 1]
                    )
                    local_actions = actions[:, current_local_start:target_tubelet_idx + 1]
                    local_intrinsics = (
                        intrinsics[:, current_local_start:target_tubelet_idx + 1]
                        if use_intrinsics else None
                    )
                    return local_states, local_actions, local_intrinsics

                def forward_predictor(step_context_latents, target_tubelet_idx):
                    """Backward-compatible wrapper over the explicit helper."""
                    local_states, local_actions, local_intrinsics = _get_local_trajectory(
                        step_context_latents,
                        target_tubelet_idx,
                    )
                    return forward_predictor_with_trajectory(
                        step_context_latents,
                        local_states,
                        local_actions,
                        local_intrinsics,
                        rollout_step=max(0, target_tubelet_idx - len(step_context_latents)),
                        target_depth=(
                            da3_depths[:, target_tubelet_idx]
                            if da3_warp_enabled and da3_depths is not None
                            else None
                        ),
                    )

                def loss_fn(
                    preds,
                    h_tgt,
                    sample_weights=None,
                    token_weights=None,
                    warped_context_raw=None,
                    unseen_boundary_mask=None,
                ):
                    """Per-layer hierarchical L1 (or ``loss_exp`` power) loss.

                    Args:
                        preds: ``(B, HW, D_concat)``.
                        h_tgt: ``(B, HW, D_concat)``.
                        sample_weights: Optional ``(B,)`` tensor of per-sample
                            loss weights. Must have ``mean == 1`` (or close to
                            it) to preserve the overall loss scale; see
                            ``_motion_sample_weights``. Broadcast to
                            ``(B, 1, 1)`` and multiplied into the per-element
                            error before the final mean. ``None`` recovers the
                            unweighted baseline exactly.
                        warped_context_raw: Optional ``(B, HW, D_concat)``.
                            When ``residual_target`` is enabled, the loss is
                            computed in residual space. If the prediction has
                            already been reconstructed as ``warp + delta``, we
                            subtract the warp from both prediction and target.
                    """
                    if residual_target and warped_context_raw is not None:
                        warp = warped_context_raw.to(
                            device=preds.device,
                            dtype=preds.dtype,
                        )
                        if residual_target_reconstruct:
                            preds = preds - warp
                        h_tgt = h_tgt - warp.to(dtype=h_tgt.dtype)
                    embed_dim = h_tgt.shape[-1] // n_hierarchical_layers
                    pred_chunks = preds.split(embed_dim, dim=-1)
                    h_chunks = h_tgt.split(embed_dim, dim=-1)
                    loss_pred = preds.new_zeros(())
                    w = sample_weights.view(-1, 1, 1) if sample_weights is not None else None
                    tw = None
                    if token_weights is not None:
                        tw = token_weights.to(device=preds.device, dtype=preds.dtype)
                        tw = tw.reshape(tw.shape[0], -1, 1)
                        tw = tw / tw.mean().clamp_min(1.0e-6)
                    if unseen_boundary_mask is not None:
                        unseen_boundary_mask = unseen_boundary_mask.to(
                            device=preds.device,
                            dtype=preds.dtype,
                        )
                    for layer_idx, (p, t) in enumerate(zip(pred_chunks, h_chunks)):
                        t = t.detach()
                        if normalize_reps:
                            p = F.layer_norm(p, (p.size(-1),))
                            t = F.layer_norm(t, (t.size(-1),))
                        if per_layer_balance:
                            # See comment in eval `_sample_latent_loss`. This
                            # keeps each hierarchical layer's gradient on a
                            # comparable scale when LN is disabled.
                            scale = t.detach().std().clamp_min(1.0e-6)
                        else:
                            scale = 1.0
                        per_elem = torch.abs(p - t) ** loss_exp / loss_exp / scale
                        if intermediate_supervision_enabled and unseen_boundary_mask is not None:
                            if layer_idx in shallow_layer_indices:
                                per_elem = per_elem * (
                                    1.0
                                    + (intermediate_shallow_weight - 1.0)
                                    * unseen_boundary_mask
                                )
                            elif layer_idx in deep_layer_indices:
                                per_elem = per_elem * (
                                    1.0
                                    + (intermediate_deep_weight - 1.0)
                                    * unseen_boundary_mask
                                )
                        if w is not None:
                            per_elem = per_elem * w
                        if tw is not None:
                            per_elem = per_elem * tw
                        loss_pred = loss_pred + per_elem.mean()
                    loss_pred = loss_pred / n_hierarchical_layers
                    return loss_pred

                # Current scheduled-sampling probability. Linearly warms up
                # from 0 -> scheduled_sampling_prob over
                # scheduled_sampling_warmup_epochs; stays at the target after.
                # warmup=0 short-circuits to the fixed target p.
                if scheduled_sampling_warmup_epochs > 0:
                    _ss_ramp = min(
                        1.0,
                        float(epoch) / float(scheduled_sampling_warmup_epochs),
                    )
                    _ss_prob_now = scheduled_sampling_prob * _ss_ramp
                else:
                    _ss_prob_now = scheduled_sampling_prob

                if rollout_context_mix_enabled:
                    if epoch < rollout_context_mix_start_epoch:
                        _ctx_mix_progress_now = 0.0
                    elif epoch >= rollout_context_mix_ramp_end_epoch:
                        _ctx_mix_progress_now = 1.0
                    else:
                        _ctx_mix_progress_now = (
                            float(epoch - rollout_context_mix_start_epoch)
                            / float(max(1, rollout_context_mix_ramp_end_epoch - rollout_context_mix_start_epoch))
                        )
                    _ctx_mix_lambda_now = rollout_context_mix_max_lambda * _ctx_mix_progress_now
                    _ctx_mix_beta_a_now = (
                        rollout_context_mix_beta_start_a
                        + _ctx_mix_progress_now
                        * (rollout_context_mix_beta_end_a - rollout_context_mix_beta_start_a)
                    )
                    _ctx_mix_beta_b_now = (
                        rollout_context_mix_beta_start_b
                        + _ctx_mix_progress_now
                        * (rollout_context_mix_beta_end_b - rollout_context_mix_beta_start_b)
                    )
                else:
                    _ctx_mix_progress_now = 1.0
                    _ctx_mix_lambda_now = 1.0
                    _ctx_mix_beta_a_now = None
                    _ctx_mix_beta_b_now = None

                # Phase R cycle weight: computed once per epoch above the
                # iter loop (it's a function of ``epoch`` only). Variable
                # ``_cycle_weight_now`` is in scope here.

                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    step_losses = []
                    delta_losses = []
                    rollout_context_latents = list(context_latents)
                    prev_target_latents = None
                    fsq_losses = []
                    sigreg_losses = []
                    # Phase C: corrector closed-loop match + denoise losses
                    # (TCR design §5.4). Lists are populated only when the
                    # corrector is enabled; aggregation happens after the
                    # rollout loop below.
                    corrector_step_losses = []
                    corrector_denoise_losses = []
                    corrector_layer_stats = None  # captured at rollout_step=0 only
                    completion_losses = []
                    completion_region_stats = []
                    patchgan_generator_losses = []
                    patchgan_discriminator_losses = []
                    patchgan_diag_stats = []
                    warp_diag_stats = []    # P0.6: depth_std per rollout step
                    layer_loss_stats = []  # P0.5: per-layer L1 per rollout step
                    # Phase R: K=1 cycle loss + drift_from_prev diagnostic
                    # (TCR design section 3.4 / 3.6). Both populated only at
                    # rollout_step=0 within the loop, since the pilot is K=1.
                    cycle_step_value = None         # tensor; aggregated below
                    drift_from_prev_value = None    # python float; logged via meter
                    # Phase R' group-composition consistency. Computed once
                    # per train_step at rollout_step==0, when we still have
                    # access to the (start = target_tubelet_idx - 1,
                    # start+1, start+2) triplet inside the same trajectory.
                    composition_step_value = None  # tensor; aggregated below
                    for rollout_step in range(effective_rollout_steps):
                        target_tubelet_idx = local_start + k_ctx + rollout_step
                        local_states_step, local_actions_step, local_intrinsics_step = _get_local_trajectory(
                            rollout_context_latents,
                            target_tubelet_idx,
                        )
                        preds_next, delta_preds_next, fsq_logits_next, warped_ctx_raw = forward_predictor_with_trajectory(
                            rollout_context_latents,
                            local_states_step,
                            local_actions_step,
                            local_intrinsics_step,
                            rollout_step=rollout_step,
                            target_depth=(
                                da3_depths[:, target_tubelet_idx]
                                if da3_warp_enabled and da3_depths is not None
                                else None
                            ),
                        )
                        if target_slot_mode == "canvas_first":
                            _canvas_diag = getattr(predictor_ref, "_last_canvas_diag", None)
                            if _canvas_diag is not None:
                                canvas_valid_meter.update(float(_canvas_diag["valid_frac"]))
                                canvas_oov_meter.update(float(_canvas_diag["oov_frac"]))
                                canvas_boundary_meter.update(float(_canvas_diag["boundary_frac"]))
                                canvas_beta_valid_meter.update(float(_canvas_diag["beta_valid"]))
                                canvas_beta_boundary_meter.update(float(_canvas_diag["beta_boundary"]))
                                canvas_beta_oov_meter.update(float(_canvas_diag["beta_oov"]))
                                canvas_input_norm_ratio_meter.update(float(_canvas_diag["input_norm_ratio"]))
                                canvas_warp_norm_meter.update(float(_canvas_diag.get("canvas_warp_norm", _canvas_diag["warp_norm"])))
                                canvas_mask_token_norm_meter.update(float(_canvas_diag["mask_token_norm"]))
                                canvas_mask_embed_norm_meter.update(float(_canvas_diag["mask_embed_norm"]))
                                canvas_type_embed_norm_meter.update(float(_canvas_diag["type_embed_norm"]))
                                canvas_block_mask_frac_meter.update(float(_canvas_diag.get("block_mask_frac", 0.0)))
                        h_tgt_this = h_target_tubelets[target_tubelet_idx]
                        _incoming_action_idx = max(0, target_tubelet_idx - 1)
                        _incoming_action = actions[:, _incoming_action_idx]
                        _outgoing_action = actions[:, target_tubelet_idx]
                        with torch.no_grad():
                            _t_in = _incoming_action[..., :3]
                            _t_out = _outgoing_action[..., :3]
                            _t_in_norm = _t_in.norm(dim=-1)
                            _t_delta = (_t_in - _t_out).norm(dim=-1)
                            _q_in = F.normalize(_incoming_action[..., 3:7], dim=-1)
                            _q_out = F.normalize(_outgoing_action[..., 3:7], dim=-1)
                            _q_dot = (_q_in * _q_out).sum(dim=-1).abs().clamp(max=1.0)
                            action_in_norm_meter.update(
                                float(_t_in_norm.mean().item())
                            )
                            action_out_norm_meter.update(
                                float(_t_out.norm(dim=-1).mean().item())
                            )
                            action_trans_cos_meter.update(
                                float(F.cosine_similarity(_t_in, _t_out, dim=-1, eps=1.0e-6).mean().item())
                            )
                            action_trans_delta_meter.update(float(_t_delta.mean().item()))
                            action_trans_ratio_meter.update(
                                float((_t_delta.mean() / _t_in_norm.mean().clamp_min(1.0e-6)).item())
                            )
                            action_rot_delta_meter.update(
                                float((2.0 * torch.acos(_q_dot)).mean().item())
                            )
                        unseen_boundary_mask = None
                        completion_masks = None
                        if intermediate_supervision_enabled:
                            unseen_boundary_mask = _rotation_unseen_boundary_mask(
                                predictor_ref,
                                local_states_step,
                                local_intrinsics_step,
                                HW,
                                preds_next.dtype,
                                intermediate_unseen_boundary_band,
                            )
                        if completion_enabled:
                            if completion_mask_source == "da3" and completion_depths is not None:
                                src_idx = max(0, target_tubelet_idx - 1)
                                completion_masks = _depth_visibility_region_masks(
                                    predictor_ref,
                                    local_states_step,
                                    local_intrinsics_step,
                                    completion_depths[:, src_idx],
                                    completion_depths[:, target_tubelet_idx],
                                    HW,
                                    preds_next.dtype,
                                    completion_depth_tau,
                                    completion_boundary_dilate,
                                )
                            else:
                                rot_mask = _rotation_unseen_boundary_mask(
                                    predictor_ref,
                                    local_states_step,
                                    local_intrinsics_step,
                                    HW,
                                    preds_next.dtype,
                                    intermediate_unseen_boundary_band,
                                )
                                if rot_mask is not None:
                                    boundary = rot_mask.to(device=preds_next.device, dtype=preds_next.dtype).clamp(0.0, 1.0)
                                    completion_masks = {
                                        "visible": 1.0 - boundary,
                                        "boundary": boundary,
                                        "disoccluded": torch.zeros_like(boundary),
                                    }

                        patchgan_masks = None
                        if latent_patchgan_enabled:
                            if (
                                completion_masks is not None
                                and completion_enabled
                                and completion_mask_source == latent_patchgan_mask_source
                            ):
                                patchgan_masks = completion_masks
                            elif latent_patchgan_mask_source == "da3" and completion_depths is not None:
                                src_idx = max(0, target_tubelet_idx - 1)
                                patchgan_masks = _depth_visibility_region_masks(
                                    predictor_ref,
                                    local_states_step,
                                    local_intrinsics_step,
                                    completion_depths[:, src_idx],
                                    completion_depths[:, target_tubelet_idx],
                                    HW,
                                    preds_next.dtype,
                                    completion_depth_tau,
                                    completion_boundary_dilate,
                                )
                            elif latent_patchgan_mask_source == "rotation":
                                rot_mask = _rotation_unseen_boundary_mask(
                                    predictor_ref,
                                    local_states_step,
                                    local_intrinsics_step,
                                    HW,
                                    preds_next.dtype,
                                    intermediate_unseen_boundary_band,
                                )
                                if rot_mask is not None:
                                    boundary = rot_mask.to(
                                        device=preds_next.device,
                                        dtype=preds_next.dtype,
                                    ).clamp(0.0, 1.0)
                                    patchgan_masks = {
                                        "visible": 1.0 - boundary,
                                        "boundary": boundary,
                                        "disoccluded": torch.zeros_like(boundary),
                                    }

                        # Motion weighting: the *incoming* action into the
                        # target step is actions[:, target_tubelet_idx - 1]
                        # (dataset convention: actions[t] = t → t+1; the last
                        # slot is zero-padded). Using target_tubelet_idx would
                        # read the outgoing (zero) action, not the motion the
                        # predictor must resolve.
                        if motion_weighted:
                            sample_w = _motion_sample_weights(
                                _incoming_action,
                                mode=motion_weight_mode,
                                alpha=motion_weight_alpha,
                            ).detach()
                        else:
                            sample_w = None

                        # 1.4: build pred_final — the blended prediction used for
                        # the main loss, self-feed, corrector, and cycle rollout.
                        #
                        # When completion_in_rollout=True and the completion head
                        # is available with a region mask, we compute comp_preds
                        # once here and blend:
                        #   pred_final = raw*(1-M) + comp*M
                        # where M = (disoccluded + boundary_weight*boundary).clamp(0,1).
                        #
                        # When the flag is off (default), pred_final == preds_next
                        # and the completion head remains a side-branch auxiliary
                        # loss only (backward-compatible).
                        #
                        # comp_preds_cached is set here if we compute it early so
                        # the completion-loss block below can reuse it without a
                        # second forward pass through completion_refine.
                        comp_preds_cached = None
                        comp_weights_cached = None
                        pred_final = preds_next
                        _alpha_now = 0.0  # initialised here; overwritten below when in_rollout
                        # Ramped alpha for E6: 0 → blend_alpha over ramp_epochs.
                        if completion_blend_alpha_ramp_epochs > 0:
                            _alpha_now = completion_blend_alpha * min(
                                1.0, epoch / max(1, completion_blend_alpha_ramp_epochs)
                            )
                        else:
                            _alpha_now = completion_blend_alpha
                        # Delayed self-feed for E7: use preds_next (not pred_final)
                        # for rollout context until selffeed_delay_epochs is reached.
                        _selffeed_pred_final = (
                            completion_in_rollout
                            and epoch >= completion_selffeed_delay_epochs
                        )
                        if (
                            completion_in_rollout
                            and use_posthoc_completion_blend
                            and completion_enabled
                            and use_completion_head
                            and completion_masks is not None
                            and hasattr(predictor_ref, "complete_predictions")
                        ):
                            # Build real mask_feats from OOV side-channel.
                            # Channel order: [valid, oov, boundary, disoccluded].
                            _oov_raw = getattr(predictor_ref, "_last_warp_oov_mask", None)
                            if _oov_raw is not None:
                                _oov_flat = _oov_raw.reshape(
                                    _oov_raw.shape[0], -1
                                ).float().to(device=preds_next.device, dtype=preds_next.dtype)
                                _oov_flat = _oov_flat.unsqueeze(-1)  # (B, HW, 1)
                            else:
                                _oov_flat = preds_next.new_zeros(preds_next.shape[0], preds_next.shape[1], 1)
                            _valid_flat = 1.0 - _oov_flat
                            _bnd_flat = completion_masks["boundary"].to(
                                device=preds_next.device, dtype=preds_next.dtype
                            ).reshape(preds_next.shape[0], -1, 1)
                            _dis_flat = completion_masks["disoccluded"].to(
                                device=preds_next.device, dtype=preds_next.dtype
                            ).reshape(preds_next.shape[0], -1, 1)
                            _mask_feats = torch.cat(
                                [_valid_flat, _oov_flat, _bnd_flat, _dis_flat], dim=-1
                            )  # (B, HW, 4)
                            # Build action_embed from incoming raw action (B, action_dim).
                            _in_act = _incoming_action.to(
                                device=preds_next.device, dtype=preds_next.dtype
                            )
                            _action_embed = _in_act
                            comp_preds_cached = predictor_ref.complete_predictions(
                                preds_next,
                                warped_context_raw=warped_ctx_raw,
                                mask_feats=_mask_feats,
                                action_embed=_action_embed,
                            )
                            _oov_mask = _oov_flat.squeeze(-1)
                            _bnd_mask = _bnd_flat.squeeze(-1)
                            if completion_use_region_blend_alpha:
                                M = (_oov_mask + _bnd_mask).clamp(0.0, 1.0)
                                _alpha_map = (
                                    completion_oov_blend_alpha * _oov_mask
                                    + completion_boundary_blend_alpha * _bnd_mask
                                ).clamp(0.0, max(completion_oov_blend_alpha, completion_boundary_blend_alpha))
                            else:
                                M = (_oov_mask + 0.25 * _bnd_mask).clamp(0.0, 1.0)
                                _alpha_map = _alpha_now * M
                            comp_weights_cached = M
                            _delta = comp_preds_cached - preds_next
                            pred_final = preds_next + _alpha_map.unsqueeze(-1) * _delta
                            # Birth diagnostics (detached, no grad).
                            with torch.no_grad():
                                birth_delta_norm_meter.update(
                                    float(_delta.abs().mean().item()))
                                birth_action_norm_meter.update(
                                    float(_action_embed.norm(dim=-1).mean().item()))
                                birth_mask_norm_meter.update(
                                    float(_mask_feats.abs().mean().item()))
                                birth_M_mean_meter.update(float(M.mean().item()))
                                birth_oov_mean_meter.update(
                                    float(_oov_flat.mean().item()))
                                birth_bnd_mean_meter.update(
                                    float(_bnd_flat.mean().item()))

                        main_step_loss = loss_fn(
                            pred_final,
                            h_tgt_this,
                            sample_weights=sample_w,
                            warped_context_raw=warped_ctx_raw,
                            unseen_boundary_mask=unseen_boundary_mask,
                        )
                        if (
                            residual_target
                            and residual_full_loss_weight > 0.0
                            and warped_ctx_raw is not None
                        ):
                            main_step_loss = main_step_loss + residual_full_loss_weight * loss_fn(
                                pred_final,
                                h_tgt_this,
                                sample_weights=sample_w,
                                unseen_boundary_mask=unseen_boundary_mask,
                            )
                        step_losses.append(main_step_loss)
                        if (
                            latent_patchgan_enabled
                            and latent_patchgan is not None
                            and patchgan_masks is not None
                        ):
                            _reveal_mask = patchgan_masks["disoccluded"].to(
                                device=pred_final.device,
                                dtype=pred_final.dtype,
                            ).reshape(pred_final.shape[0], -1, 1)
                            _reveal_count_local = _reveal_mask.sum().detach()
                            _reveal_count_global = _reveal_count_local
                            if dist.is_available() and dist.is_initialized():
                                _reveal_count_global = _reveal_count_local.clone()
                                dist.all_reduce(_reveal_count_global, op=dist.ReduceOp.SUM)
                            _has_reveal_global = bool(_reveal_count_global.item() > 0.0)
                            _grid_size = (
                                int(getattr(predictor_ref, "grid_height", crop_size // patch_size)),
                                int(getattr(predictor_ref, "grid_width", crop_size // patch_size)),
                            )
                            if _latent_patchgan_weight_now > 0.0 and _has_reveal_global:
                                _patchgan_scorer = (
                                    latent_patchgan.module
                                    if hasattr(latent_patchgan, "module")
                                    else latent_patchgan
                                )
                                _set_module_requires_grad(_patchgan_scorer, False)
                                _patchgan_was_training = _patchgan_scorer.training
                                _patchgan_scorer.eval()
                                _fake_logits_g = _patchgan_scorer(pred_final, grid_size=_grid_size)
                                if _patchgan_was_training:
                                    _patchgan_scorer.train()
                                patchgan_generator_losses.append(
                                    latent_patchgan_hinge_generator_loss(
                                        _fake_logits_g,
                                        _reveal_mask,
                                    )
                                )
                            if (
                                latent_patchgan_train_discriminator
                                and latent_patchgan_discriminator_weight > 0.0
                                and epoch >= latent_patchgan_start_epoch
                                and _has_reveal_global
                            ):
                                _set_module_requires_grad(latent_patchgan, True)
                                _real_logits_d = latent_patchgan(
                                    h_tgt_this.detach(),
                                    grid_size=_grid_size,
                                )
                                _fake_logits_d = latent_patchgan(
                                    pred_final.detach(),
                                    grid_size=_grid_size,
                                )
                                patchgan_discriminator_losses.append(
                                    latent_patchgan_discriminator_weight
                                    * latent_patchgan_hinge_discriminator_loss(
                                        _real_logits_d,
                                        _fake_logits_d,
                                        _reveal_mask,
                                    )
                                )
                            with torch.no_grad():
                                _M_pg = _reveal_mask.squeeze(-1)
                                _den_pg_raw = _M_pg.sum()
                                _has_reveal_local = bool(_den_pg_raw.item() > 0.0)
                                _den_pg = _den_pg_raw.clamp_min(1.0e-6)
                                _pred_norm_tok = pred_final.detach().norm(dim=-1)
                                _tgt_norm_tok = h_tgt_this.detach().norm(dim=-1)
                                _pred_norm = (_pred_norm_tok * _M_pg).sum().div(_den_pg)
                                _tgt_norm = (_tgt_norm_tok * _M_pg).sum().div(_den_pg)
                                patchgan_diag_stats.append(
                                    {
                                        "reveal_frac": float(_M_pg.mean().item()),
                                        "reveal_active": float(_has_reveal_local),
                                        "pred_norm": (
                                            float(_pred_norm.item())
                                            if _has_reveal_local
                                            else None
                                        ),
                                        "tgt_norm": (
                                            float(_tgt_norm.item())
                                            if _has_reveal_local
                                            else None
                                        ),
                                        "norm_ratio": (
                                            float((_pred_norm / _tgt_norm.clamp_min(1.0e-6)).item())
                                            if _has_reveal_local
                                            else None
                                        ),
                                    }
                                )
                        if (
                            completion_enabled
                            and use_completion_head
                            and completion_masks is not None
                            and hasattr(predictor_ref, "complete_predictions")
                        ):
                            # Always call complete_predictions to keep parameter
                            # usage static across iterations (DDP compatibility).
                            # Reuse cached result from in_rollout path if available.
                            if comp_weights_cached is not None:
                                comp_weights = comp_weights_cached
                                comp_preds = comp_preds_cached
                            else:
                                # Aux-only path: build real mask_feats + raw action.
                                _oov_raw_aux = getattr(predictor_ref, "_last_warp_oov_mask", None)
                                if _oov_raw_aux is not None:
                                    _oov_aux = _oov_raw_aux.reshape(
                                        _oov_raw_aux.shape[0], -1
                                    ).float().to(
                                        device=preds_next.device, dtype=preds_next.dtype
                                    ).unsqueeze(-1)
                                else:
                                    _oov_aux = preds_next.new_zeros(
                                        preds_next.shape[0], preds_next.shape[1], 1
                                    )
                                _bnd_aux = completion_masks["boundary"].to(
                                    device=preds_next.device, dtype=preds_next.dtype
                                ).reshape(preds_next.shape[0], -1).unsqueeze(-1)
                                _dis_aux = completion_masks["disoccluded"].to(
                                    device=preds_next.device, dtype=preds_next.dtype
                                ).reshape(preds_next.shape[0], -1).unsqueeze(-1)
                                _mask_feats_aux = torch.cat(
                                    [1.0 - _oov_aux, _oov_aux, _bnd_aux, _dis_aux], dim=-1
                                )
                                _in_act_aux = _incoming_action.to(
                                    device=preds_next.device, dtype=preds_next.dtype
                                )
                                comp_preds = predictor_ref.complete_predictions(
                                    preds_next,
                                    warped_context_raw=warped_ctx_raw,
                                    mask_feats=_mask_feats_aux,
                                    action_embed=_in_act_aux,
                                )
                                # OOV-centric aux weight (consistent with in_rollout M).
                                comp_weights = (
                                    _oov_aux.squeeze(-1)
                                    + completion_boundary_weight * _bnd_aux.squeeze(-1)
                                ).clamp(0.0, 1.0).to(
                                    device=preds_next.device, dtype=preds_next.dtype
                                )
                                # Birth diagnostics for aux path.
                                with torch.no_grad():
                                    _delta_aux = comp_preds - preds_next
                                    birth_delta_norm_meter.update(
                                        float(_delta_aux.abs().mean().item()))
                                    birth_action_norm_meter.update(
                                        float(_in_act_aux.norm(dim=-1).mean().item()))
                                    birth_mask_norm_meter.update(
                                        float(_mask_feats_aux.abs().mean().item()))
                                    birth_M_mean_meter.update(
                                        float(comp_weights.mean().item()))
                                    birth_oov_mean_meter.update(
                                        float(_oov_aux.mean().item()))
                                    birth_bnd_mean_meter.update(
                                        float(_bnd_aux.mean().item()))
                            completion_losses.append(
                                loss_fn(
                                    comp_preds,
                                    h_tgt_this,
                                    sample_weights=sample_w,
                                    token_weights=comp_weights,
                                )
                            )
                            with torch.no_grad():
                                _M = comp_weights.to(device=preds_next.device, dtype=preds_next.dtype).reshape(
                                    preds_next.shape[0], -1
                                )
                                _den = _M.sum().clamp_min(1.0e-6)
                                _raw_tok = (preds_next.detach() - h_tgt_this.detach()).abs().mean(dim=-1)
                                _comp_tok = (comp_preds.detach() - h_tgt_this.detach()).abs().mean(dim=-1)
                                _final_tok = (pred_final.detach() - h_tgt_this.detach()).abs().mean(dim=-1)
                                birth_raw_err_meter.update(float((_raw_tok * _M).sum().div(_den).item()))
                                birth_comp_err_meter.update(float((_comp_tok * _M).sum().div(_den).item()))
                                birth_final_err_meter.update(float((_final_tok * _M).sum().div(_den).item()))
                                _D = preds_next.shape[-1] // n_hierarchical_layers
                                _raw_layers = preds_next.detach().split(_D, dim=-1)
                                _comp_layers = comp_preds.detach().split(_D, dim=-1)
                                _final_layers = pred_final.detach().split(_D, dim=-1)
                                _tgt_layers = h_tgt_this.detach().split(_D, dim=-1)
                                for _li, (_rp, _cp, _fp, _tt) in enumerate(zip(
                                    _raw_layers, _comp_layers, _final_layers, _tgt_layers
                                )):
                                    _r = (_rp - _tt).abs().mean(dim=-1)
                                    _c = (_cp - _tt).abs().mean(dim=-1)
                                    _f = (_fp - _tt).abs().mean(dim=-1)
                                    birth_raw_layer_meters[_li].update(float((_r * _M).sum().div(_den).item()))
                                    birth_comp_layer_meters[_li].update(float((_c * _M).sum().div(_den).item()))
                                    birth_final_layer_meters[_li].update(float((_f * _M).sum().div(_den).item()))
                                completion_region_stats.append(
                                    {
                                        "visible": float(completion_masks["visible"].mean().detach().item()),
                                        "boundary": float(completion_masks["boundary"].mean().detach().item()),
                                        "disoccluded": float(completion_masks["disoccluded"].mean().detach().item()),
                                    }
                                )

                        # P0.5: per-layer prediction L1 diagnostic (no grad,
                        # cheap — just split the concat dim into n_layers).
                        with torch.no_grad():
                            if n_hierarchical_layers > 1:
                                _L = n_hierarchical_layers
                                _D = preds_next.shape[-1] // _L
                                _p = preds_next.detach().reshape(-1, _L, _D)
                                _t = h_tgt_this.detach().reshape(-1, _L, _D)
                                _ll = (_p - _t).abs().mean(dim=(0, 2))  # (_L,)
                                layer_loss_stats.append(
                                    [float(_ll[i].item()) for i in range(_L)]
                                )

                        # P0.6: depth_std + depth_cv diagnostics.
                        with torch.no_grad():
                            _td = getattr(predictor_ref, "_last_target_depth", None)
                            if _td is not None:
                                _td_flat = _td.detach().flatten(1)  # (B, HW)
                                _dstd = float(_td_flat.std(dim=1).mean().item())
                                _dmean = float(_td_flat.mean(dim=1).mean().item())
                                _dcv = _dstd / max(_dmean, 1e-6)
                                warp_diag_stats.append({"depth_std": _dstd, "depth_cv": _dcv})

                        if use_delta_head and delta_preds_next is not None:
                            if prev_target_latents is None:
                                ctx_last_idx = target_tubelet_idx - 1
                                h_current = h_target_tubelets[ctx_last_idx]
                            else:
                                h_current = prev_target_latents
                            delta_target = h_tgt_this - h_current.detach()
                            # Same motion weighting: the delta loss is *literally*
                            # supervising how much things changed, so amplifying
                            # high-motion samples is even more natural here.
                            delta_losses.append(loss_fn(delta_preds_next, delta_target, sample_weights=sample_w))
                        prev_target_latents = h_tgt_this

                        if use_fsq_head and fsq_logits_next is not None and fsq_module is not None:
                            # Encode target features once per step; the frozen
                            # FSQ runs under no_grad, so this is cheap and the
                            # CE signal only trains the predictor's fsq_head.
                            with torch.no_grad():
                                tgt_idx = fsq_module.encode(h_tgt_this.detach())
                            fsq_losses.append(
                                categorical_loss(fsq_logits_next.float(), tgt_idx)
                            )

                        # SIGReg: regularize the predicted latent toward the
                        # GT-latent distribution (pre-whitened so the target
                        # is N(0, 1) per LeWM). Applied to the raw predictor
                        # output -- the same tensor that may become the next
                        # rollout step's self-fed input -- so the closed-loop
                        # input distribution at eval time has been shaped at
                        # every training step.
                        if sigreg_weight > 0.0:
                            sigreg_losses.append(
                                _sigreg_match_to_target(
                                    preds_next,
                                    h_tgt_this,
                                    n_directions=sigreg_n_directions,
                                    n_layers=n_hierarchical_layers,
                                )
                            )

                        # Phase R — reversibility cycle loss (TCR design
                        # section 3.4) and drift_from_prev diagnostic
                        # (section 3.6). Pilot is K=1, so we only run the
                        # cycle at rollout_step == 0; the cycle uses the
                        # forward prediction at this step as the seed and
                        # asks the predictor to apply the SE(3) inverse of
                        # actions[t_end - 1] to recover the previous frame's
                        # GT latent.
                        if rollout_step == 0 and target_tubelet_idx >= 1:
                            # drift_from_prev: how much the forward
                            # prediction differs from the most recent
                            # context latent. Identity collapse signature
                            # is this dropping >30%; logged via meter.
                            with torch.no_grad():
                                _ctx_last_for_drift = h_target_tubelets[
                                    target_tubelet_idx - 1
                                ].detach()
                                drift_from_prev_value = float(
                                    (preds_next.detach() - _ctx_last_for_drift)
                                    .abs().mean().item()
                                )

                            if cycle_enabled and _cycle_weight_now > 0.0:
                                t_end = target_tubelet_idx
                                # Backward action: apply T_(t_end -> t_end-1)
                                # = invert(actions[t_end - 1]). The action at
                                # t_end - 1 in the dataset convention encodes
                                # T_(next_to_cur) for the (t_end-1, t_end)
                                # pair, mapping cam(t_end) coords into
                                # cam(t_end-1) coords. Going backward in
                                # time we want the inverse of that.
                                _action_fwd_step = actions[:, t_end - 1].to(
                                    dtype=preds_next.dtype
                                )
                                _action_bwd_step = _invert_action_se3(_action_fwd_step)
                                _zero_action = torch.zeros_like(_action_bwd_step)
                                local_actions_bwd = torch.stack(
                                    [_action_bwd_step, _zero_action], dim=1
                                )                              # (B, 2, 7)

                                # State window: {t_end, t_end - 1} re-canonicalized
                                # to slot 0 (= original t_end). Per design
                                # doc section 3.3 the helper handles the
                                # re-canonicalization automatically.
                                _states_raw_bwd = torch.stack(
                                    [states[:, t_end], states[:, t_end - 1]],
                                    dim=1,
                                ).to(dtype=preds_next.dtype)
                                local_states_bwd = _canonicalize_states(_states_raw_bwd)

                                local_intrinsics_bwd = (
                                    torch.stack(
                                        [intrinsics[:, t_end], intrinsics[:, t_end - 1]],
                                        dim=1,
                                    ).to(dtype=preds_next.dtype)
                                    if use_intrinsics else None
                                )

                                # Backward seed. Predicted-seed (default) is
                                # the paper claim: forward prediction must be
                                # invertible. Teacher-forced seed is a
                                # diagnostic-only ablation (R4).
                                if cycle_seed_mode == "predicted":
                                    # Use pred_final: the completion-blended
                                    # latent when completion_in_rollout=True,
                                    # else == preds_next (backward-compatible).
                                    z_seed = (
                                        pred_final.detach()
                                        if cycle_detach_forward
                                        else pred_final
                                    )
                                else:
                                    z_seed = h_target_tubelets[t_end].detach()

                                preds_back, _, _, _ = forward_predictor_with_trajectory(
                                    [z_seed],
                                    local_states_bwd,
                                    local_actions_bwd,
                                    local_intrinsics_bwd,
                                    target_depth=(
                                        da3_depths[:, t_end - 1]
                                        if da3_warp_enabled and da3_depths is not None
                                        else None
                                    ),
                                )
                                # Supervise against the GT target latent at
                                # the previous tubelet.
                                z_target_back = h_target_tubelets[t_end - 1].detach()
                                if cycle_visibility_mask == "adaptive":
                                    # R2 ablation: bottom-50% forward-error
                                    # tokens. Computed from preds_next vs
                                    # h_tgt_this (the forward prediction's
                                    # error). Inline masked L1 (mean over
                                    # all dims to keep scale comparable to
                                    # loss_fn for K=1 horizon=1).
                                    with torch.no_grad():
                                        _fwd_err = (
                                            preds_next.detach() - h_tgt_this.detach()
                                        ).abs().mean(dim=-1)              # (B, HW)
                                        _thresh = _fwd_err.median(
                                            dim=-1, keepdim=True
                                        ).values
                                        _cmask = (_fwd_err < _thresh).float().unsqueeze(-1)
                                    _per_elem = (preds_back - z_target_back).abs() * _cmask
                                    cycle_step_value = (
                                        _per_elem.sum() / _cmask.sum().clamp_min(1.0) / preds_back.shape[-1]
                                    )
                                else:
                                    cycle_step_value = loss_fn(preds_back, z_target_back)

                        # Phase R' — group-composition consistency loss.
                        # Build the composed action over the (start, start+2)
                        # window where start = target_tubelet_idx - 1 (the
                        # last context tubelet, GT teacher-forced). Run the
                        # predictor with this composed action over a 2-slot
                        # state window canonicalized to slot 0, mirroring
                        # the cycle backward-call construction. Anchor the
                        # composed-path output against z_gt at start+2 (the
                        # supervised anchor), and optionally against the
                        # sequential 2-step prediction (path-path consistency).
                        if (
                            rollout_step == 0
                            and composition_enabled
                            and _composition_weight_now > 0.0
                            and target_tubelet_idx >= 1
                            and (target_tubelet_idx + 1) < total_tubelets
                        ):
                            comp_start = target_tubelet_idx - 1
                            comp_end = target_tubelet_idx + 1  # = start + 2

                            # Composed action: applies actions[start] then
                            # actions[start+1] in matrix form
                            # M_a @ M_b == _compose_action_se3(a, b).
                            # Verified by
                            # ``test_two_action_composition_advances_state_two_steps``.
                            _action_a = actions[:, comp_start].to(
                                dtype=preds_next.dtype
                            )
                            _action_b = actions[:, comp_start + 1].to(
                                dtype=preds_next.dtype
                            )
                            _composed_action = _compose_action_se3(
                                _action_a, _action_b
                            )
                            _zero_action_co = torch.zeros_like(_composed_action)
                            local_actions_co = torch.stack(
                                [_composed_action, _zero_action_co], dim=1
                            )                                  # (B, 2, 7)

                            # State window: slot 0 = comp_start, slot 1 =
                            # comp_end, canonicalized so slot 0 is identity.
                            # The relative-pose action above is independent
                            # of the canonicalization anchor, so passing it
                            # directly is correct (see Phase R cycle for the
                            # mirror-image construction).
                            _states_raw_co = torch.stack(
                                [states[:, comp_start], states[:, comp_end]],
                                dim=1,
                            ).to(dtype=preds_next.dtype)
                            local_states_co = _canonicalize_states(_states_raw_co)

                            local_intrinsics_co = (
                                torch.stack(
                                    [intrinsics[:, comp_start],
                                     intrinsics[:, comp_end]],
                                    dim=1,
                                ).to(dtype=preds_next.dtype)
                                if use_intrinsics else None
                            )

                            # Composition seed. Teacher (default, R'-pilot)
                            # uses the GT latent at start, eliminating any
                            # confound from scheduled sampling. "predicted"
                            # is reserved for a later ablation that asks
                            # the predictor to produce a self-consistent
                            # composed prediction starting from its own
                            # forward output.
                            if composition_seed_mode == "teacher":
                                z_seed_co = h_target_tubelets[comp_start].detach()
                            else:
                                # "predicted": use the predictor's forward
                                # output for the (comp_start) tubelet. Since
                                # rollout_step=0 predicts comp_start + 1
                                # (NOT comp_start), we cannot reuse
                                # ``preds_next`` here. The cleanest predicted
                                # seed is the most recent context slot in
                                # ``rollout_context_latents`` if that slot
                                # itself originated from the predictor on a
                                # previous training step — but in this pilot
                                # we restrict to teacher seeding to avoid
                                # entanglement with scheduled sampling.
                                raise NotImplementedError(
                                    "composition_seed='predicted' is not "
                                    "supported in the R' pilot; use 'teacher'."
                                )

                            preds_composed, _, _, _ = forward_predictor_with_trajectory(
                                [z_seed_co],
                                local_states_co,
                                local_actions_co,
                                local_intrinsics_co,
                                target_depth=(
                                    da3_depths[:, comp_end]
                                    if da3_warp_enabled and da3_depths is not None
                                    else None
                                ),
                            )

                            # Aggregate the supervised anchor + optional
                            # path-path consistency into a single tensor.
                            # The path-path term is detached on the RHS by
                            # default to prevent the trivial mean-collapse
                            # solution (both branches agree on a degenerate
                            # constant). The sequential 2-step path is a
                            # teacher-seeded chain of two single-step
                            # predictor calls; this is bit-equivalent to
                            # running the existing rollout from a clean
                            # GT seed (which the live rollout does not do
                            # because of scheduled sampling) and adds at
                            # most two extra forward passes per train step.
                            _composition_terms = []
                            if composition_supervised_anchor:
                                z_gt_t2 = h_target_tubelets[comp_end].detach()
                                _composition_terms.append(
                                    loss_fn(preds_composed, z_gt_t2)
                                )
                            if composition_path_path_weight > 0.0:
                                # Sequential 2-step from teacher-forced
                                # start. Step 1: comp_start -> comp_start+1.
                                _states_seq_1 = _canonicalize_states(
                                    torch.stack(
                                        [states[:, comp_start],
                                         states[:, comp_start + 1]],
                                        dim=1,
                                    ).to(dtype=preds_next.dtype)
                                )
                                _zero_action_seq_1 = torch.zeros_like(_action_a)
                                _actions_seq_1 = torch.stack(
                                    [_action_a, _zero_action_seq_1], dim=1
                                )
                                _intr_seq_1 = (
                                    torch.stack(
                                        [intrinsics[:, comp_start],
                                         intrinsics[:, comp_start + 1]],
                                        dim=1,
                                    ).to(dtype=preds_next.dtype)
                                    if use_intrinsics else None
                                )
                                _z_mid_pp, _, _, _ = forward_predictor_with_trajectory(
                                    [z_seed_co],
                                    _states_seq_1,
                                    _actions_seq_1,
                                    _intr_seq_1,
                                    target_depth=(
                                        da3_depths[:, comp_start + 1]
                                        if da3_warp_enabled and da3_depths is not None
                                        else None
                                    ),
                                )
                                # Step 2: comp_start+1 -> comp_end (= +2).
                                _states_seq_2 = _canonicalize_states(
                                    torch.stack(
                                        [states[:, comp_start + 1],
                                         states[:, comp_end]],
                                        dim=1,
                                    ).to(dtype=preds_next.dtype)
                                )
                                _zero_action_seq_2 = torch.zeros_like(_action_b)
                                _actions_seq_2 = torch.stack(
                                    [_action_b, _zero_action_seq_2], dim=1
                                )
                                _intr_seq_2 = (
                                    torch.stack(
                                        [intrinsics[:, comp_start + 1],
                                         intrinsics[:, comp_end]],
                                        dim=1,
                                    ).to(dtype=preds_next.dtype)
                                    if use_intrinsics else None
                                )
                                preds_seq_2step, _, _, _ = forward_predictor_with_trajectory(
                                    [_z_mid_pp],
                                    _states_seq_2,
                                    _actions_seq_2,
                                    _intr_seq_2,
                                    target_depth=(
                                        da3_depths[:, comp_end]
                                        if da3_warp_enabled and da3_depths is not None
                                        else None
                                    ),
                                )
                                _rhs = (
                                    preds_seq_2step.detach()
                                    if composition_detach_rhs
                                    else preds_seq_2step
                                )
                                _composition_terms.append(
                                    composition_path_path_weight
                                    * loss_fn(preds_composed, _rhs)
                                )
                            if _composition_terms:
                                composition_step_value = sum(_composition_terms)
                            else:
                                # Both terms off — should be impossible due
                                # to the config validator above; keep the
                                # branch for robustness.
                                composition_step_value = preds_next.new_zeros(())

                        # Phase C — latent corrector hook (TCR design §5.4).
                        # When the corrector is enabled, run it on the
                        # predicted latent. The corrected output is what
                        # feeds the next rollout step. The corrector is
                        # supervised by the closed-loop match loss against
                        # the GT target latent at this step.
                        if corrector is not None:
                            B_sz = preds_next.shape[0]
                            _step_idx = torch.full(
                                (B_sz,), rollout_step,
                                device=preds_next.device, dtype=torch.long,
                            )
                            _pose_mag = torch.linalg.norm(
                                _incoming_action[:, :3], dim=-1
                            )
                            # C1 frozen-predictor: detach to fully cut graph;
                            # only corrector params receive gradient.
                            # C2 joint: keep graph, predictor learns to
                            # produce correctable latents.
                            # Use pred_final (completion-blended when
                            # completion_in_rollout=True, else == preds_next).
                            _corr_input = (
                                pred_final.detach()
                                if corrector_mode == "frozen_predictor"
                                else pred_final
                            )
                            z_corr = corrector(_corr_input, _step_idx, _pose_mag)
                            corrector_step_losses.append(
                                loss_fn(z_corr, h_tgt_this, sample_weights=sample_w)
                            )

                            # Denoise objective: corrector should map a
                            # noisy GT latent back to clean GT. Cheap and
                            # mode-agnostic; trains the corrector on more
                            # of the manifold than just predictor outputs.
                            if (
                                corrector_denoise_weight > 0.0
                                and corrector_noise_std > 0.0
                            ):
                                with torch.no_grad():
                                    _tgt_scale_d = h_tgt_this.detach().std(
                                        dim=(0, 1), keepdim=True
                                    ).clamp_min(1.0e-6)
                                _noise_d = torch.randn_like(h_tgt_this) * (
                                    _tgt_scale_d * corrector_noise_std
                                )
                                z_noisy = h_tgt_this.detach() + _noise_d
                                z_denoised = corrector(z_noisy, _step_idx, _pose_mag)
                                corrector_denoise_losses.append(
                                    loss_fn(
                                        z_denoised, h_tgt_this.detach(),
                                        sample_weights=sample_w,
                                    )
                                )

                            # Per-layer monitoring: capture once per
                            # train_step at rollout_step=0 so logging stays
                            # cheap (one extra forward pass on rank 0 only).
                            if (
                                rollout_step == 0
                                and rank == 0
                                and (itr % log_freq == 0)
                                and corrector_layer_stats is None
                            ):
                                _corr_unwrapped = (
                                    corrector.module
                                    if hasattr(corrector, "module")
                                    else corrector
                                )
                                corrector_layer_stats = (
                                    _corr_unwrapped.per_layer_norm_stats(
                                        _corr_input.detach(), _step_idx, _pose_mag,
                                    )
                                )

                        # Scheduled sampling: choose what the *next* rollout
                        # step sees as its most-recent context slot. With
                        # probability ``_ss_prob_now`` we self-feed (append
                        # the predictor's own output, current two-step
                        # behaviour). With probability ``1 - p`` we
                        # teacher-force (append the GT target latent), which
                        # shortens the error-compounding chain by one step.
                        # Detach the GT branch so teacher-forced slots do not
                        # backprop through the target encoder at this point.
                        if _ss_prob_now >= 1.0 or random.random() < _ss_prob_now:
                            # When the corrector is on, the self-fed slot is
                            # the corrected prediction (paper claim: the
                            # corrector applies at inference time on the
                            # rollout slot). When off, fall back to pred_final
                            # Use pred_final for self-feed when in_rollout is on
                            # and the delay period has passed (E7 delayed selffeed).
                            # Before delay: self-feed preds_next to prevent instability.
                            _slot_for_selffeed = (
                                pred_final
                                if _selffeed_pred_final
                                else preds_next
                            )
                            _next_slot = z_corr if corrector is not None else _slot_for_selffeed
                            # A3 input noise injection: per-layer scale-aware
                            # Gaussian perturbation on the self-fed slot.
                            # Noise scale = rollout_noise_std * per-channel
                            # std of the GT target latent at this step, so
                            # magnitudes are comparable across hierarchical
                            # layers regardless of normalize_reps setting.
                            if rollout_noise_std > 0.0:
                                with torch.no_grad():
                                    _tgt_scale = h_tgt_this.detach().std(
                                        dim=(0, 1), keepdim=True
                                    ).clamp_min(1.0e-6)
                                _noise = torch.randn_like(_next_slot) * (
                                    _tgt_scale * rollout_noise_std
                                )
                                _next_slot = _next_slot + _noise
                            if rollout_context_mix_enabled:
                                if rollout_context_mix_mode == "beta":
                                    if _ctx_mix_progress_now <= 0.0:
                                        _ctx_mix_lambda = _next_slot.new_zeros(
                                            (_next_slot.shape[0], 1, 1)
                                        )
                                    else:
                                        with torch.amp.autocast("cuda", enabled=False):
                                            _beta_dist = torch.distributions.Beta(
                                                torch.tensor(
                                                    float(_ctx_mix_beta_a_now),
                                                    device=_next_slot.device,
                                                    dtype=torch.float32,
                                                ),
                                                torch.tensor(
                                                    float(_ctx_mix_beta_b_now),
                                                    device=_next_slot.device,
                                                    dtype=torch.float32,
                                                ),
                                            )
                                            _ctx_mix_lambda = _beta_dist.sample(
                                                (_next_slot.shape[0], 1, 1)
                                            )
                                        _ctx_mix_lambda = _ctx_mix_lambda.to(
                                            device=_next_slot.device,
                                            dtype=_next_slot.dtype,
                                        )
                                        _ctx_mix_lambda = (
                                            _ctx_mix_lambda
                                            * _ctx_mix_progress_now
                                            * rollout_context_mix_max_lambda
                                        ).clamp(0.0, 1.0)
                                else:
                                    _ctx_mix_lambda = _next_slot.new_full(
                                        (_next_slot.shape[0], 1, 1),
                                        float(_ctx_mix_lambda_now),
                                    )
                                _next_slot = (
                                    (1.0 - _ctx_mix_lambda) * h_tgt_this.detach()
                                    + _ctx_mix_lambda * _next_slot
                                )
                        else:
                            _next_slot = h_tgt_this.detach()
                        rollout_context_latents = (rollout_context_latents + [_next_slot])[-k_ctx:]

                    loss_pred = step_losses[0]
                    loss_step1 = step_losses[0]
                    loss_step2 = step_losses[1] if len(step_losses) > 1 else loss_pred.new_zeros(())
                    loss_ctx = loss_pred.new_zeros(())
                    step_weight = rollout_loss_decay
                    for step_loss in step_losses[1:]:
                        loss_ctx = loss_ctx + step_weight * step_loss
                        step_weight = step_weight * rollout_loss_decay
                    loss = loss_pred + loss_ctx

                    if use_delta_head and delta_losses:
                        loss_delta_total = sum(delta_losses) / len(delta_losses)
                        loss = loss + delta_loss_weight * loss_delta_total

                    if sigreg_weight > 0.0 and sigreg_losses:
                        loss_sigreg_total = sum(sigreg_losses) / len(sigreg_losses)
                        loss = loss + sigreg_weight * loss_sigreg_total

                    if use_fsq_head and fsq_losses:
                        loss_fsq_total = sum(fsq_losses) / len(fsq_losses)
                        loss = loss + fsq_loss_weight * loss_fsq_total

                    loss_completion = loss.new_zeros(())
                    completion_region_means = None
                    if completion_losses:
                        loss_completion = sum(completion_losses) / len(completion_losses)
                        loss = loss + completion_loss_weight * loss_completion
                        if completion_region_stats:
                            completion_region_means = {
                                key: float(np.mean([s[key] for s in completion_region_stats]))
                                for key in ("visible", "boundary", "disoccluded")
                            }

                    loss_patchgan_g = loss.new_zeros(())
                    loss_patchgan_d = loss.new_zeros(())
                    patchgan_diag_means = None
                    if patchgan_generator_losses:
                        loss_patchgan_g = (
                            sum(patchgan_generator_losses)
                            / len(patchgan_generator_losses)
                        )
                        loss = loss + _latent_patchgan_weight_now * loss_patchgan_g
                    if patchgan_discriminator_losses:
                        loss_patchgan_d = (
                            sum(patchgan_discriminator_losses)
                            / len(patchgan_discriminator_losses)
                        )
                    if patchgan_diag_stats:
                        _patchgan_norm_stats = [
                            s for s in patchgan_diag_stats
                            if s["pred_norm"] is not None
                        ]
                        patchgan_diag_means = {
                            "reveal_frac": float(np.mean([s["reveal_frac"] for s in patchgan_diag_stats])),
                            "reveal_active": float(np.mean([s["reveal_active"] for s in patchgan_diag_stats])),
                            "pred_norm": (
                                float(np.mean([s["pred_norm"] for s in _patchgan_norm_stats]))
                                if _patchgan_norm_stats
                                else None
                            ),
                            "tgt_norm": (
                                float(np.mean([s["tgt_norm"] for s in _patchgan_norm_stats]))
                                if _patchgan_norm_stats
                                else None
                            ),
                            "norm_ratio": (
                                float(np.mean([s["norm_ratio"] for s in _patchgan_norm_stats]))
                                if _patchgan_norm_stats
                                else None
                            ),
                        }

                    # P0.5/P0.6: aggregate per-layer and depth diagnostics.
                    warp_diag_means = None
                    layer_loss_means = None
                    if warp_diag_stats:
                        warp_diag_means = {
                            "depth_std": float(np.mean([s["depth_std"] for s in warp_diag_stats])),
                            "depth_cv": float(np.mean([s["depth_cv"] for s in warp_diag_stats])),
                        }
                    if layer_loss_stats:
                        _n = len(layer_loss_stats[0])
                        layer_loss_means = [
                            float(np.mean([s[i] for s in layer_loss_stats]))
                            for i in range(_n)
                        ]

                    # Phase C — aggregate corrector closed-loop match +
                    # denoise losses. Closed-loop match is the primary
                    # corrector signal; denoise is the regularizer.
                    loss_corr_step = loss.new_zeros(())
                    loss_corr_denoise = loss.new_zeros(())
                    if corrector_step_losses:
                        loss_corr_step = sum(corrector_step_losses) / len(corrector_step_losses)
                        # Weight 1.0 for closed-loop match (same scale as
                        # standard prediction loss); paper's primary signal.
                        loss = loss + loss_corr_step
                    if corrector_denoise_losses:
                        loss_corr_denoise = sum(corrector_denoise_losses) / len(corrector_denoise_losses)
                        loss = loss + corrector_denoise_weight * loss_corr_denoise

                    # Phase R — add reversibility cycle loss to the total
                    # objective. Pilot is K=1 so cycle_step_value is the
                    # full cycle term; the ramped weight handles warmup.
                    loss_cycle = loss.new_zeros(())
                    if cycle_step_value is not None:
                        loss_cycle = cycle_step_value
                        if _cycle_weight_now > 0.0:
                            loss = loss + _cycle_weight_now * loss_cycle

                    # Phase R' — add group-composition consistency loss.
                    # The supervised anchor (and optional path-path term)
                    # was computed at rollout_step=0 inside the loop above.
                    # Here we just fold the precomputed scalar into the
                    # total objective using the ramped weight.
                    loss_composition = loss.new_zeros(())
                    if composition_step_value is not None:
                        loss_composition = composition_step_value
                        if _composition_weight_now > 0.0:
                            loss = loss + _composition_weight_now * loss_composition

                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if mixed_precision:
                    scaler.step(optimizer)
                else:
                    optimizer.step()
                optimizer.zero_grad()
                if latent_patchgan_optimizer is not None:
                    # Keep the discriminator update separate from the
                    # predictor update: D sees detached real/fake latents,
                    # while the generator loss only used D as a frozen scoring
                    # function.
                    latent_patchgan_optimizer.zero_grad(set_to_none=True)
                    if patchgan_discriminator_losses:
                        if mixed_precision:
                            scaler.scale(loss_patchgan_d).backward()
                            scaler.unscale_(latent_patchgan_optimizer)
                            scaler.step(latent_patchgan_optimizer)
                        else:
                            loss_patchgan_d.backward()
                            latent_patchgan_optimizer.step()
                    latent_patchgan_optimizer.zero_grad(set_to_none=True)
                if mixed_precision:
                    scaler.update()

                _cycle_loss_value = (
                    float(loss_cycle.item())
                    if isinstance(loss_cycle, torch.Tensor) and loss_cycle.numel() > 0
                    else 0.0
                )
                _composition_loss_value = (
                    float(loss_composition.item())
                    if isinstance(loss_composition, torch.Tensor)
                    and loss_composition.numel() > 0
                    else 0.0
                )
                _completion_loss_value = (
                    float(loss_completion.item())
                    if isinstance(loss_completion, torch.Tensor)
                    and loss_completion.numel() > 0
                    else 0.0
                )
                _patchgan_g_value = (
                    float(loss_patchgan_g.item())
                    if isinstance(loss_patchgan_g, torch.Tensor)
                    and loss_patchgan_g.numel() > 0
                    else 0.0
                )
                _patchgan_d_value = (
                    float(loss_patchgan_d.item())
                    if isinstance(loss_patchgan_d, torch.Tensor)
                    and loss_patchgan_d.numel() > 0
                    else 0.0
                )
                _drift_value = (
                    float(drift_from_prev_value)
                    if drift_from_prev_value is not None
                    else float("nan")
                )
                return (
                    loss.item(),
                    loss_pred.item(),
                    loss_ctx.item(),
                    loss_step1.item(),
                    loss_step2.item(),
                    _new_lr,
                    _new_wd,
                    loss_corr_step.item(),
                    loss_corr_denoise.item(),
                    corrector_layer_stats,
                    _cycle_loss_value,
                    _composition_loss_value,
                    _drift_value,
                    _completion_loss_value,
                    _patchgan_g_value,
                    _patchgan_d_value,
                    patchgan_diag_means,
                    completion_region_means,
                    warp_diag_means,
                    layer_loss_means,
                )

            (
                loss, loss_pred, loss_ctx, loss_step1, loss_step2, _new_lr, _new_wd,
                _loss_corr_step, _loss_corr_denoise, _corr_layer_stats,
                _loss_cycle, _loss_composition, _drift_from_prev,
                _loss_completion, _loss_patchgan_g, _loss_patchgan_d,
                _patchgan_diag_means, _completion_region_means,
                _warp_diag_means, _layer_loss_means,
            ), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            loss_pred_meter.update(loss_pred)
            loss_ctx_meter.update(loss_ctx)
            loss_step1_meter.update(loss_step1)
            loss_step2_meter.update(loss_step2)
            if corrector is not None:
                loss_corr_step_meter.update(_loss_corr_step)
                loss_corr_denoise_meter.update(_loss_corr_denoise)
                if _corr_layer_stats is not None:
                    last_corrector_layer_stats = _corr_layer_stats
            if cycle_enabled:
                # Track cycle loss only after warmup ramps it in; before
                # that the value is computed (when seed mode allows) but
                # not contributing to the loss.
                loss_cycle_meter.update(_loss_cycle)
            if composition_enabled and _loss_composition > 0.0:
                # Same convention as cycle: only meter when the loss is
                # actively contributing (start_epoch reached + ramp > 0).
                loss_composition_meter.update(_loss_composition)
            if not np.isnan(_drift_from_prev):
                drift_from_prev_meter.update(_drift_from_prev)
            if completion_enabled and _loss_completion > 0.0:
                loss_completion_meter.update(_loss_completion)
            if latent_patchgan_enabled:
                if _loss_patchgan_g != 0.0:
                    patchgan_g_meter.update(_loss_patchgan_g)
                if _loss_patchgan_d != 0.0:
                    patchgan_d_meter.update(_loss_patchgan_d)
                if _patchgan_diag_means is not None:
                    patchgan_reveal_frac_meter.update(_patchgan_diag_means["reveal_frac"])
                    if _patchgan_diag_means["pred_norm"] is not None:
                        patchgan_pred_norm_meter.update(_patchgan_diag_means["pred_norm"])
                        patchgan_tgt_norm_meter.update(_patchgan_diag_means["tgt_norm"])
                        patchgan_norm_ratio_meter.update(_patchgan_diag_means["norm_ratio"])
            if _completion_region_means is not None:
                completion_visible_meter.update(_completion_region_means["visible"])
                completion_boundary_meter.update(_completion_region_means["boundary"])
                completion_disoccluded_meter.update(_completion_region_means["disoccluded"])
            # P0.5/P0.6: update per-layer and depth diagnostics.
            if _warp_diag_means is not None:
                depth_std_meter.update(_warp_diag_means["depth_std"])
                depth_cv_meter.update(_warp_diag_means["depth_cv"])
            if _layer_loss_means is not None:
                for _li, _lv in enumerate(_layer_loss_means):
                    if _li < len(layer_loss_meters):
                        layer_loss_meters[_li].update(_lv)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, loss_pred, loss_ctx, loss_step1, loss_step2,
                    iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    _corr_log = ""
                    if corrector is not None:
                        _corr_log = (
                            " [corr_step: %.4f corr_denoise: %.4f]"
                            % (loss_corr_step_meter.avg, loss_corr_denoise_meter.avg)
                        )
                    _cycle_log = ""
                    if cycle_enabled:
                        _cycle_log = (
                            " [cycle: %.4f w: %.3f drift: %.4f]"
                            % (
                                loss_cycle_meter.avg,
                                _cycle_weight_now,
                                drift_from_prev_meter.avg,
                            )
                        )
                    elif drift_from_prev_meter.count > 0:
                        # Always surface the drift diagnostic when it has
                        # values, even if the cycle loss is disabled.
                        _cycle_log = " [drift: %.4f]" % drift_from_prev_meter.avg
                    _completion_log = ""
                    if completion_enabled and loss_completion_meter.count > 0:
                        _completion_log = (
                            " [comp: %.4f vis/bnd/dis: %.3f/%.3f/%.3f]"
                            % (
                                loss_completion_meter.avg,
                                completion_visible_meter.avg,
                                completion_boundary_meter.avg,
                                completion_disoccluded_meter.avg,
                            )
                        )
                    _patchgan_log = ""
                    if latent_patchgan_enabled and patchgan_reveal_frac_meter.count > 0:
                        _patchgan_log = (
                            " [patchgan g/d: %.4f/%.4f w: %.4f reveal: %.3f norm p/t/r: %.3f/%.3f/%.3f]"
                            % (
                                patchgan_g_meter.avg,
                                patchgan_d_meter.avg,
                                _latent_patchgan_weight_now,
                                patchgan_reveal_frac_meter.avg,
                                patchgan_pred_norm_meter.avg,
                                patchgan_tgt_norm_meter.avg,
                                patchgan_norm_ratio_meter.avg,
                            )
                        )
                    _birth_log = ""
                    if completion_enabled and birth_delta_norm_meter.count > 0:
                        _birth_raw_layers = "/".join("%.4f" % m.avg for m in birth_raw_layer_meters)
                        _birth_comp_layers = "/".join("%.4f" % m.avg for m in birth_comp_layer_meters)
                        _birth_final_layers = "/".join("%.4f" % m.avg for m in birth_final_layer_meters)
                        _birth_log = (
                            " [birth_delta: %.4f act: %.4f msk: %.4f M: %.3f oov: %.3f bnd: %.3f act_in/out: %.4f/%.4f act_cos: %.3f act_delta: %.4f act_ratio: %.3f act_rot: %.4f in_roll: %s alpha: %.3f birth_err raw/comp/final: %.4f/%.4f/%.4f birth_L raw:%s comp:%s final:%s]"
                            % (
                                birth_delta_norm_meter.avg,
                                birth_action_norm_meter.avg,
                                birth_mask_norm_meter.avg,
                                birth_M_mean_meter.avg,
                                birth_oov_mean_meter.avg,
                                birth_bnd_mean_meter.avg,
                                action_in_norm_meter.avg,
                                action_out_norm_meter.avg,
                                action_trans_cos_meter.avg,
                                action_trans_delta_meter.avg,
                                action_trans_ratio_meter.avg,
                                action_rot_delta_meter.avg,
                                str(completion_in_rollout),
                                (completion_blend_alpha * min(1.0, epoch / max(1, completion_blend_alpha_ramp_epochs))
                                 if completion_blend_alpha_ramp_epochs > 0
                                 else completion_blend_alpha) if completion_in_rollout else 0.0,
                                birth_raw_err_meter.avg,
                                birth_comp_err_meter.avg,
                                birth_final_err_meter.avg,
                                _birth_raw_layers,
                                _birth_comp_layers,
                                _birth_final_layers,
                            )
                        )
                    _diag_log = ""
                    if depth_std_meter.count > 0:
                        _diag_log += " [depth_std: %.3f cv: %.3f]" % (
                            depth_std_meter.avg, depth_cv_meter.avg
                        )
                    if layer_loss_meters[0].count > 0:
                        _ll_str = "/".join("%.4f" % m.avg for m in layer_loss_meters)
                        _diag_log += " [layer_L1: %s]" % _ll_str
                    if target_slot_mode == "canvas_first" and canvas_valid_meter.count > 0:
                        _diag_log += (
                            " [canvas valid/oov/bnd: %.3f/%.3f/%.3f beta: %.3f/%.3f/%.3f norm_ratio: %.3f warp/mask/mskemb/typeemb: %.3f/%.3f/%.3f/%.3f blk: %.3f]"
                            % (
                                canvas_valid_meter.avg,
                                canvas_oov_meter.avg,
                                canvas_boundary_meter.avg,
                                canvas_beta_valid_meter.avg,
                                canvas_beta_boundary_meter.avg,
                                canvas_beta_oov_meter.avg,
                                canvas_input_norm_ratio_meter.avg,
                                canvas_warp_norm_meter.avg,
                                canvas_mask_token_norm_meter.avg,
                                canvas_mask_embed_norm_meter.avg,
                                canvas_type_embed_norm_meter.avg,
                                canvas_block_mask_frac_meter.avg,
                            )
                        )
                    if getattr(predictor_ref, "camera_ucpe_enabled", False):
                        _ucpe_gammas = [
                            "L%d:%.4f" % (i, predictor_ref.camera_ucpe_branches[str(i)].gamma.item())
                            for i in predictor_ref.camera_ucpe_apply_layers
                        ]
                        _ucpe_grad_norms = [
                            "L%d:%.4f" % (
                                i,
                                predictor_ref.camera_ucpe_branches[str(i)].gamma.grad.item()
                                if predictor_ref.camera_ucpe_branches[str(i)].gamma.grad is not None
                                else float("nan"),
                            )
                            for i in predictor_ref.camera_ucpe_apply_layers
                        ]
                        _diag_log += " [ucpe_gamma: %s ucpe_gamma_grad: %s]" % (
                            " ".join(_ucpe_gammas),
                            " ".join(_ucpe_grad_norms),
                        )
                    logger.info(
                        "[%d, %5d] loss: %.3f [pred: %.3f ctx: %.3f step1: %.3f step2: %.3f]%s%s%s%s%s%s "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                        % (
                            epoch + 1, itr,
                            loss_meter.avg, loss_pred_meter.avg, loss_ctx_meter.avg,
                            loss_step1_meter.avg, loss_step2_meter.avg,
                            _corr_log,
                            _cycle_log,
                            _completion_log,
                            _patchgan_log,
                            _birth_log,
                            _diag_log,
                            _new_wd, _new_lr,
                            torch.cuda.max_memory_allocated() / 1024.0**2,
                            iter_time_meter.avg,
                            gpu_time_meter.avg,
                            data_elapsed_time_meter.avg,
                        )
                    )

            log_stats()
            assert not np.isnan(loss), "loss is nan"

        logger.info("avg. loss %.3f" % loss_meter.avg)

        training_stats["loss"].append(loss_meter.avg)
        training_stats["loss_pred"].append(loss_pred_meter.avg)
        training_stats["loss_ctx"].append(loss_ctx_meter.avg)
        training_stats["loss_step1"].append(loss_step1_meter.avg)
        training_stats["loss_step2"].append(loss_step2_meter.avg)
        training_stats["lr"].append(scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else _new_lr)
        training_stats["wd"].append(_new_wd)
        training_stats["iter_ms"].append(iter_time_meter.avg)
        training_stats["gpu_ms"].append(gpu_time_meter.avg)
        training_stats["mem_gb"].append(torch.cuda.max_memory_allocated() / 1024.0**3)
        if cycle_enabled:
            training_stats["cycle_l1_h1"].append(loss_cycle_meter.avg)
            training_stats["cycle_weight"].append(_cycle_weight_now)
        if composition_enabled:
            training_stats["composition_l1"].append(loss_composition_meter.avg)
            training_stats["composition_weight"].append(_composition_weight_now)
        if completion_enabled:
            training_stats["completion_l1"].append(loss_completion_meter.avg)
            training_stats["completion_visible_frac"].append(completion_visible_meter.avg)
            training_stats["completion_boundary_frac"].append(completion_boundary_meter.avg)
            training_stats["completion_disoccluded_frac"].append(completion_disoccluded_meter.avg)
        if latent_patchgan_enabled:
            training_stats["patchgan_g"].append(patchgan_g_meter.avg)
            training_stats["patchgan_d"].append(patchgan_d_meter.avg)
            training_stats["patchgan_weight"].append(_latent_patchgan_weight_now)
            training_stats["patchgan_reveal_frac"].append(patchgan_reveal_frac_meter.avg)
            training_stats["patchgan_pred_norm"].append(patchgan_pred_norm_meter.avg)
            training_stats["patchgan_tgt_norm"].append(patchgan_tgt_norm_meter.avg)
            training_stats["patchgan_norm_ratio"].append(patchgan_norm_ratio_meter.avg)
        if target_slot_mode == "canvas_first":
            training_stats["canvas_valid_frac"].append(canvas_valid_meter.avg)
            training_stats["canvas_oov_frac"].append(canvas_oov_meter.avg)
            training_stats["canvas_boundary_frac"].append(canvas_boundary_meter.avg)
            training_stats["canvas_beta_valid"].append(canvas_beta_valid_meter.avg)
            training_stats["canvas_beta_boundary"].append(canvas_beta_boundary_meter.avg)
            training_stats["canvas_beta_oov"].append(canvas_beta_oov_meter.avg)
            training_stats["canvas_input_norm_ratio"].append(canvas_input_norm_ratio_meter.avg)
        if drift_from_prev_meter.count > 0:
            training_stats["drift_from_prev"].append(drift_from_prev_meter.avg)
        if corrector is not None:
            training_stats["loss_corr_step"].append(loss_corr_step_meter.avg)
            training_stats["loss_corr_denoise"].append(loss_corr_denoise_meter.avg)
            if last_corrector_layer_stats is not None:
                for _l in range(n_hierarchical_layers):
                    training_stats[f"corr_input_norm_l{_l}"].append(
                        last_corrector_layer_stats.get(f"input_norm_l{_l}", float("nan"))
                    )
                    training_stats[f"corr_output_norm_l{_l}"].append(
                        last_corrector_layer_stats.get(f"output_norm_l{_l}", float("nan"))
                    )
                    training_stats[f"corr_delta_norm_l{_l}"].append(
                        last_corrector_layer_stats.get(f"delta_norm_l{_l}", float("nan"))
                    )
                    training_stats[f"corr_scale_l{_l}"].append(
                        last_corrector_layer_stats.get(f"scale_l{_l}", float("nan"))
                    )

        final_epoch = epoch == (num_epochs - 1)
        periodic_checkpoint_epoch = save_every_freq > 0 and (epoch + 1) % save_every_freq == 0
        if checkpoint_saving_enabled and epoch > 0 and (periodic_checkpoint_epoch or (final_epoch and save_final_checkpoint)):
            checkpoint_saved = False
            checkpoint_path = None
            if final_epoch and save_final_checkpoint:
                checkpoint_path = final_path
                checkpoint_saved = save_checkpoint(epoch + 1, checkpoint_path)
            elif periodic_checkpoint_epoch:
                checkpoint_path = os.path.join(scratch_folder, f"e{epoch + 1}.pt")
                checkpoint_saved = save_checkpoint(epoch + 1, checkpoint_path)
            checkpoint_saved = _broadcast_bool_from_rank0(checkpoint_saved)

            if checkpoint_saved:
                _prune_checkpoint_outputs(checkpoint_path)
                if checkpoint_eval_enabled:
                    run_checkpoint_evaluation(epoch)
            elif periodic_checkpoint_epoch or (final_epoch and save_final_checkpoint):
                logger.warning("Keeping previous checkpoint because the new checkpoint save did not succeed.")
                logger.warning("Skipping checkpoint evaluation because checkpoint save did not succeed.")
            _safe_dist_barrier()

    _shutdown_distributed()
