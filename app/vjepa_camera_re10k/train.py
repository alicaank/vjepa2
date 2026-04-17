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
import os
import glob
import copy
import gc
import random
import shutil
import sys
import tempfile
import time

import numpy as np
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from app.vjepa_camera_re10k.utils import init_opt, init_video_model, load_checkpoint, load_pretrained
from src.utils.distributed import init_distributed
from src.utils.logging import AverageMeter, CSVLogger, get_logger, gpu_timer
from src.training.visualization import visualize_pca_features, plot_training_curves

log_timings = True
log_freq = 100
CHECKPOINT_FREQ = 10
VIZ_FREQ = 10
GARBAGE_COLLECT_ITR_FREQ = 50
MIN_FREE_BYTES_FOR_FULL_CHECKPOINT = 6 * 1024**3

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _quat_conjugate(quat):
    return torch.cat([-quat[..., :3], quat[..., 3:4]], dim=-1)


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
    use_delta_head = bool(cfgs_model.get("use_delta_head", False))
    delta_loss_weight = float(cfgs_model.get("delta_loss_weight", 0.25))
    use_residual_head = bool(cfgs_model.get("use_residual_head", False))
    residual_head_depth = int(cfgs_model.get("residual_head_depth", 2))
    residual_head_ratio = float(cfgs_model.get("residual_head_ratio", 2.0))

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
    tubelet_size = cfgs_data.get("tubelet_size", 2)
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
    rollout_train_steps = max(1, int(cfgs_loss.get("rollout_train_steps", 1)))
    rollout_loss_decay = float(cfgs_loss.get("rollout_loss_decay", 0.5))
    rollout_train_steps = min(rollout_train_steps, max_valid_ctx_tubelets)
    max_rollout_ctx_tubelets = max(1, total_tubelets - rollout_train_steps)
    if rollout_train_steps > 1 and max_ctx_tubelets > max_rollout_ctx_tubelets:
        logger.warning(
            f"rollout_train_steps={rollout_train_steps} with total_tubelets={total_tubelets} "
            f"caps effective training context to at most {max_rollout_ctx_tubelets} tubelets."
        )

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
        use_delta_head=use_delta_head,
        use_residual_head=use_residual_head,
        residual_head_depth=residual_head_depth,
        residual_head_ratio=residual_head_ratio,
    )
    target_encoder = copy.deepcopy(encoder)

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
    optimizer, scaler, scheduler, wd_scheduler = init_opt(
        encoder=encoder,
        predictor=predictor,
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
    if use_ddp:
        encoder = DistributedDataParallel(encoder, static_graph=True)
        predictor = DistributedDataParallel(predictor, static_graph=True)
        target_encoder = DistributedDataParallel(target_encoder)
        logger.info("Wrapped encoder/predictor/target_encoder with DistributedDataParallel.")
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

    # -- load pretrained encoder weights
    encoder, predictor, target_encoder = load_pretrained(
        r_path=p_file,
        encoder=encoder,
        predictor=predictor,
        target_encoder=target_encoder,
        context_encoder_key=context_encoder_key,
        target_encoder_key=target_encoder_key,
        load_predictor=load_predictor,
        load_encoder=load_encoder,
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
        pretrained_cache_path = None
        if isinstance(p_file, str) and p_file:
            if os.path.isfile(p_file):
                pretrained_cache_path = p_file
            elif p_file.startswith(("https://", "http://")):
                torch_home = os.environ.get("TORCH_HOME", os.path.expanduser("~/.cache/torch"))
                pretrained_cache_path = os.path.join(torch_home, "hub", "checkpoints", os.path.basename(p_file))
        if pretrained_cache_path is not None:
            resume_abs = os.path.abspath(resume_path) if resume_path is not None else None
            if resume_abs is None or os.path.abspath(pretrained_cache_path) != resume_abs:
                _remove_file(pretrained_cache_path)

    def _build_full_checkpoint_payload(epoch):
        return {
            "encoder": encoder.state_dict(),
            "predictor": predictor.state_dict(),
            "opt": optimizer.state_dict(),
            "scaler": None if scaler is None else scaler.state_dict(),
            "target_encoder": target_encoder.state_dict(),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "checkpoint_type": "full",
        }

    def _build_weights_only_payload(epoch, *, include_target_encoder):
        payload = {
            "encoder": _compact_state_dict(encoder.state_dict()),
            "predictor": _compact_state_dict(predictor.state_dict()),
            "epoch": epoch,
            "loss": loss_meter.avg,
            "batch_size": batch_size,
            "world_size": world_size,
            "lr": lr,
            "checkpoint_type": "weights_only_bf16",
        }
        if include_target_encoder:
            payload["target_encoder"] = _compact_state_dict(target_encoder.state_dict())
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
        if checkpoint_save_mode == "full" or free_bytes >= MIN_FREE_BYTES_FOR_FULL_CHECKPOINT:
            if _write_checkpoint_payload(_build_full_checkpoint_payload(epoch), path, tag="full"):
                return True
            logger.warning("Full checkpoint save failed; falling back to weights-only checkpoint.")
        else:
            logger.warning(
                f"Only {free_bytes / 1024.0**3:.2f} GiB free in {checkpoint_stage_dir}; "
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
                loss_pred = loss_pred + (
                    torch.mean(torch.abs(p - t) ** loss_exp) / loss_exp
                )
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
            preds_v, _, _delta_v = _pred(
                h_in,
                v_actions[:, :target_idx + 1],
                v_states[:, :target_idx + 1],
                intrinsics=v_intrinsics[:, :target_idx + 1] if use_intrinsics else None,
            )
            return {
                "imgs": v_imgs,
                "preds_next": preds_v[:, -HW:, :],
                "h_tgt_eval": h_tgt_eval,
                "h_tgt_last": h_tgt_last,
                "h_pred_last": preds_v[0, -HW:, -layer_dim:],
                "rel_action": v_actions[0, target_idx - 1],
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

        if pca_outputs_written > 0:
            _prune_pca_outputs(epoch)
            logger.info(f"PCA visualisations saved to {pca_vis_dir}")
        else:
            logger.warning("No PCA visualisations were written; keeping previous PCA outputs.")

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
        iter_time_meter = AverageMeter()
        gpu_time_meter = AverageMeter()
        data_elapsed_time_meter = AverageMeter()

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

                def forward_predictor(step_context_latents, target_tubelet_idx):
                    h_context_step = torch.cat(step_context_latents, dim=1)
                    B_sz = h_context_step.shape[0]
                    current_local_start = target_tubelet_idx - len(step_context_latents)
                    local_states = _canonicalize_states(states[:, current_local_start:target_tubelet_idx + 1])
                    local_actions = actions[:, current_local_start:target_tubelet_idx + 1]
                    local_intrinsics = (
                        intrinsics[:, current_local_start:target_tubelet_idx + 1] if use_intrinsics else None
                    )
                    future_tokens = predictor_ref.future_mask_token.to(
                        device=h_context_step.device,
                        dtype=h_context_step.dtype,
                    )
                    future_tokens = future_tokens.expand(B_sz, HW, -1)
                    h_predictor_input = torch.cat([h_context_step, future_tokens], dim=1)
                    preds, _, delta_preds = predictor(
                        h_predictor_input,
                        local_actions,
                        local_states,
                        intrinsics=local_intrinsics,
                    )
                    if delta_preds is not None:
                        return preds[:, -HW:, :], delta_preds[:, -HW:, :]
                    return preds[:, -HW:, :], None

                def loss_fn(preds, h_tgt):
                    embed_dim = h_tgt.shape[-1] // n_hierarchical_layers
                    pred_chunks = preds.split(embed_dim, dim=-1)
                    h_chunks = h_tgt.split(embed_dim, dim=-1)
                    loss_pred = preds.new_zeros(())
                    for p, t in zip(pred_chunks, h_chunks):
                        t = t.detach()
                        if normalize_reps:
                            p = F.layer_norm(p, (p.size(-1),))
                            t = F.layer_norm(t, (t.size(-1),))
                        loss_pred = loss_pred + (
                            torch.mean(torch.abs(p - t) ** loss_exp) / loss_exp
                        )
                    loss_pred = loss_pred / n_hierarchical_layers
                    return loss_pred

                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    step_losses = []
                    delta_losses = []
                    rollout_context_latents = list(context_latents)
                    prev_target_latents = None
                    for rollout_step in range(effective_rollout_steps):
                        target_tubelet_idx = local_start + k_ctx + rollout_step
                        preds_next, delta_preds_next = forward_predictor(
                            rollout_context_latents,
                            target_tubelet_idx,
                        )
                        h_tgt_this = h_target_tubelets[target_tubelet_idx]
                        step_losses.append(loss_fn(preds_next, h_tgt_this))

                        if use_delta_head and delta_preds_next is not None:
                            if prev_target_latents is None:
                                ctx_last_idx = target_tubelet_idx - 1
                                h_current = h_target_tubelets[ctx_last_idx]
                            else:
                                h_current = prev_target_latents
                            delta_target = h_tgt_this - h_current.detach()
                            delta_losses.append(loss_fn(delta_preds_next, delta_target))
                        prev_target_latents = h_tgt_this

                        rollout_context_latents = (rollout_context_latents + [preds_next])[-k_ctx:]

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

                return (
                    loss.item(),
                    loss_pred.item(),
                    loss_ctx.item(),
                    loss_step1.item(),
                    loss_step2.item(),
                    _new_lr,
                    _new_wd,
                )

            (loss, loss_pred, loss_ctx, loss_step1, loss_step2, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            loss_pred_meter.update(loss_pred)
            loss_ctx_meter.update(loss_ctx)
            loss_step1_meter.update(loss_step1)
            loss_step2_meter.update(loss_step2)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, loss_pred, loss_ctx, loss_step1, loss_step2,
                    iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [pred: %.3f ctx: %.3f step1: %.3f step2: %.3f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                        % (
                            epoch + 1, itr,
                            loss_meter.avg, loss_pred_meter.avg, loss_ctx_meter.avg,
                            loss_step1_meter.avg, loss_step2_meter.avg,
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
