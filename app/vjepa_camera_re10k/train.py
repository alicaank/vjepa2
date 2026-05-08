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
    # Rotation-only homography warp of context-frame tokens into target grid
    # (see CameraConditionedPredictorAC._warp_context_tokens). Diagnostic
    # ablation: default off, gated by cfgs_model and/or env var.
    warp_context_latents = bool(cfgs_model.get("warp_context_latents", False))
    warp_context_padding_mode = str(cfgs_model.get("warp_context_padding_mode", "border"))
    # Path B-lite Step 1 — depth-aware projective warp via a frozen probe.
    # ``warp_mode`` selects the transport: "auto" maps to legacy behaviour
    # (rotation_only iff warp_context_latents=True else off); "rotation_only"
    # forces the legacy infinity-plane homography; "projective_probe" uses
    # the depth-probe target depth and requires ``depth_probe_checkpoint``.
    warp_mode = str(cfgs_model.get("warp_mode", "auto"))
    depth_probe_checkpoint = cfgs_model.get("depth_probe_checkpoint", None)
    if depth_probe_checkpoint is not None:
        depth_probe_checkpoint = str(depth_probe_checkpoint)

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
    if completion_enabled and not use_completion_head:
        logger.warning(
            "loss.completion.enabled=True but model.use_completion_head=False; "
            "enabling the masked region metrics but skipping completion-head loss."
        )
    if completion_enabled and completion_mask_source == "da3" and not _DEPTH_TEACHER_AVAILABLE:
        raise ImportError(
            "loss.completion.mask_source='da3' requires src.training.common.depth_teacher."
        )
    # Run B — residual target: predict z_target - warp(z_last, target)
    # instead of full z_target.  Requires warp_mode != "off" and a valid
    # depth_probe_checkpoint so the warped context is available.
    residual_target = bool(cfgs_loss.get("residual_target", False))
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
        use_delta_head=use_delta_head,
        use_residual_head=use_residual_head,
        residual_head_depth=residual_head_depth,
        residual_head_ratio=residual_head_ratio,
        use_completion_head=use_completion_head,
        completion_head_depth=completion_head_depth,
        completion_head_ratio=completion_head_ratio,
        use_fsq_head=use_fsq_head,
        fsq_total_code_axes=fsq_total_code_axes,
        fsq_levels=fsq_levels,
        warp_context_latents=warp_context_latents,
        warp_context_padding_mode=warp_context_padding_mode,
        warp_mode=warp_mode,
        depth_probe_checkpoint=depth_probe_checkpoint,
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
    if completion_enabled and completion_mask_source == "da3":
        if load_depth_teacher is None:
            raise RuntimeError("DA3 completion masks requested but depth teacher loader is unavailable.")
        completion_depth_processor, completion_depth_teacher = load_depth_teacher(
            device=device,
            model_id=completion_da3_model_id,
        )
        logger.info(
            "Completion route DA3 masks: model=%s depth_res=%d",
            completion_da3_model_id,
            completion_da3_depth_res,
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

    # Paper-faithful V-JEPA 2-AC: when the context-encoder LR scale is 0 we
    # also disable its gradients so the backward pass actually skips the
    # encoder (~30% step-time reduction + encoder-sized gradient VRAM saved).
    # Without this guard AdamW would still compute encoder grads and then
    # zero-update them — wasteful.
    context_encoder_frozen = float(enc_lr_scale) <= 0.0
    if context_encoder_frozen:
        for p in encoder.parameters():
            p.requires_grad = False
        # eval() disables dropout / stochastic-depth inside ViT blocks so
        # context features are deterministic — matches the target encoder
        # regime and guarantees context == target outputs at init (paper
        # figure's identical "frozen encoder" boxes).
        encoder.eval()
        logger.info(
            "Context encoder frozen (enc_lr_scale=%.3f <= 0): "
            "parameters set requires_grad=False and module.eval() called.",
            float(enc_lr_scale),
        )

    logger.info(
        "Loss config: normalize_reps=%s, per_layer_balance=%s, loss_exp=%.3f, "
        "rollout_train_steps=%d, motion_weighted=%s (mode=%s, alpha=%.2f), "
        "intermediate_supervision=%s.",
        normalize_reps, per_layer_balance, float(loss_exp), rollout_train_steps,
        motion_weighted, motion_weight_mode, motion_weight_alpha,
        intermediate_supervision_enabled,
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
    # MASt3R adapter loaded its own pretrained weights via
    # ``AsymmetricMASt3R.from_pretrained``, so the V-JEPA pretrain
    # checkpoint must not be re-applied to the encoder. We still let
    # ``load_pretrained`` run so it can warm-start the predictor when
    # ``load_predictor=True``; just disable the encoder side.
    _load_encoder_effective = load_encoder and (encoder_backbone != "mast3r")
    if encoder_backbone == "mast3r" and load_encoder:
        logger.info(
            "encoder_backbone=mast3r: skipping load_pretrained on encoder/target_encoder "
            "(MASt3R weights already loaded inside MASt3REncoderAdapter)."
        )
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
        if rank != 0:
            return False
        _purge_runtime_caches()
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

    logger.info("Initializing loader...")
    unsupervised_sampler.set_epoch(start_epoch)
    loader = iter(unsupervised_loader)

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
            future_tokens_v = _pred.future_mask_token.to(device=device, dtype=h_ctx.dtype).expand(
                B_v, HW, -1
            )
            h_in = torch.cat([h_ctx, future_tokens_v], dim=1)
            preds_v, _, _delta_v, _fsq_v, _warped_ctx_v = _pred(
                h_in,
                v_actions[:, :target_idx + 1],
                v_states[:, :target_idx + 1],
                intrinsics=v_intrinsics[:, :target_idx + 1] if use_intrinsics else None,
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
                "preds_next": preds_v[:, -HW:, :],
                "h_tgt_eval": h_tgt_eval,
                "h_tgt_last": h_tgt_last,
                "h_pred_last": preds_v[0, -HW:, -layer_dim:],
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
        completion_visible_meter = AverageMeter()
        completion_boundary_meter = AverageMeter()
        completion_disoccluded_meter = AverageMeter()
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
                ctx_layer_outs = encoder(ctx_clip)
                h_context = torch.cat(ctx_layer_outs, dim=-1)
                context_latents = [
                    h_context[:, tubelet_idx * HW:(tubelet_idx + 1) * HW, :]
                    for tubelet_idx in range(k_ctx)
                ]
                predictor_ref = predictor.module if hasattr(predictor, "module") else predictor
                completion_depths = None
                if completion_enabled and completion_mask_source == "da3":
                    if completion_depth_teacher is None or compute_teacher_depths_batch_with_confidence is None:
                        raise RuntimeError("DA3 completion masks requested but depth teacher is not initialized.")
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
                        completion_depths = depth_maps.reshape(B, total_tubelets, H, W)

                def forward_predictor_with_trajectory(
                    step_context_latents,
                    local_states_canon,
                    local_actions_window,
                    local_intrinsics_window,
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
                    )
                    fsq_slice = fsq_logits[:, -HW:, :, :] if fsq_logits is not None else None
                    if delta_preds is not None:
                        return preds[:, -HW:, :], delta_preds[:, -HW:, :], fsq_slice, warped_ctx_raw
                    return preds[:, -HW:, :], None, fsq_slice, warped_ctx_raw

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
                            When ``residual_target`` is enabled, ``h_tgt`` is
                            replaced by ``h_tgt - warped_context_raw`` so the
                            predictor only models the residual beyond what
                            projective geometry explains.
                    """
                    if residual_target and warped_context_raw is not None:
                        h_tgt = h_tgt - warped_context_raw.to(dtype=h_tgt.dtype)
                    embed_dim = h_tgt.shape[-1] // n_hierarchical_layers
                    pred_chunks = preds.split(embed_dim, dim=-1)
                    h_chunks = h_tgt.split(embed_dim, dim=-1)
                    loss_pred = preds.new_zeros(())
                    w = sample_weights.view(-1, 1, 1) if sample_weights is not None else None
                    tw = None
                    if token_weights is not None:
                        tw = token_weights.to(device=preds.device, dtype=preds.dtype)
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
                        )
                        h_tgt_this = h_target_tubelets[target_tubelet_idx]
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

                        # Motion weighting: action at the step being predicted
                        # is what the predictor must actually resolve. Weights
                        # are normalized to mean=1 per batch so loss scale
                        # tracks the unweighted baseline when motion is uniform.
                        if motion_weighted:
                            sample_w = _motion_sample_weights(
                                actions[:, target_tubelet_idx],
                                mode=motion_weight_mode,
                                alpha=motion_weight_alpha,
                            ).detach()
                        else:
                            sample_w = None

                        step_losses.append(
                            loss_fn(
                                preds_next,
                                h_tgt_this,
                                sample_weights=sample_w,
                                warped_context_raw=warped_ctx_raw,
                                unseen_boundary_mask=unseen_boundary_mask,
                            )
                        )
                        if (
                            completion_enabled
                            and use_completion_head
                            and completion_masks is not None
                            and hasattr(predictor_ref, "complete_predictions")
                        ):
                            # NOTE: this branch must be deterministic across
                            # iterations or DDP's reducer fires the
                            # ``ready twice`` error on ``completion_refine``
                            # parameters. The previous implementation skipped
                            # the call entirely when ``comp_weights.sum() == 0``,
                            # making the graph data-dependent — incompatible
                            # with both ``static_graph=True`` and the
                            # ``find_unused_parameters=True`` reducer when the
                            # parameter is touched in multi-step rollout.
                            #
                            # Always call ``complete_predictions`` and append
                            # the loss with the same ``comp_weights`` mask;
                            # the loss naturally contributes zero when no
                            # boundary/disoccluded tokens are present, but
                            # the parameter usage pattern stays static.
                            comp_weights = (
                                completion_masks["disoccluded"]
                                + completion_boundary_weight * completion_masks["boundary"]
                            ).to(device=preds_next.device, dtype=preds_next.dtype)
                            comp_preds = predictor_ref.complete_predictions(
                                preds_next,
                                warped_context_raw=warped_ctx_raw,
                            )
                            completion_losses.append(
                                loss_fn(
                                    comp_preds,
                                    h_tgt_this,
                                    sample_weights=sample_w,
                                    token_weights=comp_weights,
                                )
                            )
                            with torch.no_grad():
                                completion_region_stats.append(
                                    {
                                        "visible": float(completion_masks["visible"].mean().detach().item()),
                                        "boundary": float(completion_masks["boundary"].mean().detach().item()),
                                        "disoccluded": float(completion_masks["disoccluded"].mean().detach().item()),
                                    }
                                )

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
                                    z_seed = (
                                        preds_next.detach()
                                        if cycle_detach_forward
                                        else preds_next
                                    )
                                else:
                                    z_seed = h_target_tubelets[t_end].detach()

                                preds_back, _, _, _ = forward_predictor_with_trajectory(
                                    [z_seed],
                                    local_states_bwd,
                                    local_actions_bwd,
                                    local_intrinsics_bwd,
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
                                actions[:, target_tubelet_idx, :3], dim=-1
                            )
                            # C1 frozen-predictor: detach to fully cut graph;
                            # only corrector params receive gradient.
                            # C2 joint: keep graph, predictor learns to
                            # produce correctable latents.
                            _corr_input = (
                                preds_next.detach()
                                if corrector_mode == "frozen_predictor"
                                else preds_next
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
                            # rollout slot). When off, fall back to the raw
                            # predicted latent.
                            _next_slot = z_corr if corrector is not None else preds_next
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
                    scaler.update()
                else:
                    optimizer.step()
                optimizer.zero_grad()

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
                    completion_region_means,
                )

            (
                loss, loss_pred, loss_ctx, loss_step1, loss_step2, _new_lr, _new_wd,
                _loss_corr_step, _loss_corr_denoise, _corr_layer_stats,
                _loss_cycle, _loss_composition, _drift_from_prev,
                _loss_completion, _completion_region_means,
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
            if _completion_region_means is not None:
                completion_visible_meter.update(_completion_region_means["visible"])
                completion_boundary_meter.update(_completion_region_means["boundary"])
                completion_disoccluded_meter.update(_completion_region_means["disoccluded"])
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
                    logger.info(
                        "[%d, %5d] loss: %.3f [pred: %.3f ctx: %.3f step1: %.3f step2: %.3f]%s%s%s "
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

        if epoch > 0 and (epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1)):
            checkpoint_saved = False
            checkpoint_path = None
            if epoch == (num_epochs - 1):
                checkpoint_path = final_path
                checkpoint_saved = save_checkpoint(epoch + 1, checkpoint_path)
            elif save_every_freq > 0 and epoch % save_every_freq == 0:
                checkpoint_path = os.path.join(scratch_folder, f"e{epoch}.pt")
                checkpoint_saved = save_checkpoint(epoch + 1, checkpoint_path)
            checkpoint_saved = _broadcast_bool_from_rank0(checkpoint_saved)

            if checkpoint_saved:
                _prune_checkpoint_outputs(checkpoint_path)
                run_checkpoint_evaluation(epoch)
            elif epoch == (num_epochs - 1) or (save_every_freq > 0 and epoch % save_every_freq == 0):
                logger.warning("Keeping previous checkpoint because the new checkpoint save did not succeed.")
                logger.warning("Skipping checkpoint evaluation because checkpoint save did not succeed.")
            _safe_dist_barrier()

    _shutdown_distributed()
