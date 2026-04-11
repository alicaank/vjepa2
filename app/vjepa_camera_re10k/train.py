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

import os

try:
    os.environ["CUDA_VISIBLE_DEVICES"] = os.environ["SLURM_LOCALID"]
except Exception:
    pass

import copy
import gc
import random
import sys
import time

import numpy as np
import torch
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
CHECKPOINT_FREQ = 1000
VIZ_FREQ = 500
GARBAGE_COLLECT_ITR_FREQ = 50

_GLOBAL_SEED = 0
random.seed(_GLOBAL_SEED)
np.random.seed(_GLOBAL_SEED)
torch.manual_seed(_GLOBAL_SEED)
torch.backends.cudnn.benchmark = True

logger = get_logger(__name__, force=True)


def _init_re10k_loader(
    data_root,
    seq_len,
    stride,
    min_stride,
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

    # -- DATA
    cfgs_data = args.get("data")
    data_root = cfgs_data.get("data_root")
    seq_len = cfgs_data.get("seq_len", 8)
    stride = cfgs_data.get("stride", 4)
    min_stride = cfgs_data.get("min_stride", 1)
    batch_size = cfgs_data.get("batch_size")
    tubelet_size = cfgs_data.get("tubelet_size", 2)
    crop_size = cfgs_data.get("crop_size", 256)
    patch_size = cfgs_data.get("patch_size", 16)
    pin_mem = cfgs_data.get("pin_mem", True)
    num_workers = cfgs_data.get("num_workers", 8)
    persistent_workers = cfgs_data.get("persistent_workers", True)
    # Number of context tubelets shown to the context encoder.
    # The predictor must hallucinate the remaining (T_tok - n_ctx_tubelets) tubelets.
    n_ctx_tubelets = cfgs_data.get("n_ctx_tubelets", 1)
    # Manifest / cache paths (forwarded to RE10KLazySceneDataset)
    manifest_cache_dir = cfgs_data.get("manifest_cache_dir", None)
    fixed_manifest_path = cfgs_data.get("fixed_manifest_path", None)
    local_chunk_cache_dir = cfgs_data.get("local_chunk_cache_dir", None)
    local_chunk_cache_limit_gb = float(cfgs_data.get("local_chunk_cache_limit_gb", 0.0))
    eval_fixed_manifest_path = cfgs_data.get("eval_fixed_manifest_path", None)
    n_pca_scenes = int(cfgs_data.get("n_pca_scenes", 4))

    # -- LOSS
    cfgs_loss = args.get("loss")
    loss_exp = cfgs_loss.get("loss_exp", 1.0)
    normalize_reps = cfgs_loss.get("normalize_reps", True)

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
    latest_path = os.path.join(scratch_folder, "latest.pt")
    resume_path = os.path.join(scratch_folder, r_file) if r_file is not None else latest_path
    if not os.path.exists(resume_path):
        resume_path = None

    csv_logger = CSVLogger(
        log_file,
        ("%d", "epoch"),
        ("%d", "itr"),
        ("%.5f", "loss"),
        ("%.5f", "loss_pred"),
        ("%.5f", "loss_ctx"),
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
        _eval_ds = RE10KSequenceDataset(
            lazy_dataset=_eval_lazy, seq_len=seq_len, stride=stride,
            min_stride=min_stride, image_size=crop_size,
        )
        pca_loader = DataLoader(
            _eval_ds, batch_size=1, shuffle=False,
            num_workers=2, pin_memory=False,
            collate_fn=collate_re10k_sequences, drop_last=False,
        )
        logger.info(f"PCA eval loader: {len(_eval_ds)} test scenes")
    except Exception as _e:
        pca_loader = None
        logger.warning(f"PCA eval loader unavailable: {_e}")

    unsupervised_loader, unsupervised_sampler = _init_re10k_loader(
        data_root=data_root,
        seq_len=seq_len,
        stride=stride,
        min_stride=min_stride,
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

    encoder = DistributedDataParallel(encoder, static_graph=True)
    predictor = DistributedDataParallel(predictor, static_graph=True)
    target_encoder = DistributedDataParallel(target_encoder)
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

    loss_meter = AverageMeter()

    def save_checkpoint(epoch, path):
        if rank != 0:
            return
        save_dict = {
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
        }
        tmp_path = path + ".tmp"
        try:
            torch.save(save_dict, tmp_path)
            os.replace(tmp_path, path)
        except Exception as e:
            logger.info(f"Encountered exception when saving checkpoint: {e}")
            if os.path.exists(tmp_path):
                os.remove(tmp_path)

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

    # Per-epoch stats accumulator for training curves
    training_stats = {"loss": [], "loss_pred": [], "loss_ctx": [], "lr": [], "wd": [],
                      "iter_ms": [], "gpu_ms": [], "mem_gb": []}

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
                # images: [B, T, 3, H, W] → encoder expects [B, C, T, H, W]
                imgs = sample["images"].to(device, non_blocking=True)
                # Dataset yields per-frame poses (T=seq_len).
                # Predictor expects per-tubelet poses (T//tubelet_size).
                # Subsample: take one pose per tubelet (first frame of each).
                s = sample["states"].to(device, dtype=torch.float, non_blocking=True)       # [B, T, 7]
                a = sample["actions"].to(device, dtype=torch.float, non_blocking=True)      # [B, T, 7]
                k = sample["intrinsics"].to(device, dtype=torch.float, non_blocking=True)   # [B, T, 4]
                states     = s[:, ::tubelet_size, :]   # [B, T//ts, 7]
                actions    = a[:, ::tubelet_size, :]   # [B, T//ts, 7]
                intrinsics = k[:, ::tubelet_size, :]   # [B, T//ts, 4]
                return imgs, states, actions, intrinsics

            imgs, states, actions, intrinsics = load_batch()
            B, T_seq, C, H, W = imgs.shape
            data_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            if sync_gc and (itr + 1) % GARBAGE_COLLECT_ITR_FREQ == 0:
                gc.collect()

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()

                HW = (crop_size // patch_size) ** 2          # spatial tokens per tubelet
                T_tok = seq_len // tubelet_size               # total tubelets

                def _layerlist_to_h(layer_outs):
                    """list[Tensor[B, T_tok*HW, D]] → [B, T_tok*HW, n_layers*D]"""
                    if normalize_reps:
                        layer_outs = [
                            F.layer_norm(feat, (feat.size(-1),))
                            for feat in layer_outs
                        ]
                    return torch.cat(layer_outs, dim=-1)

                # ------------------------------------------------------------ #
                # 1. TARGET: frozen target_encoder sees the FULL clip
                # ------------------------------------------------------------ #
                with torch.no_grad():
                    full_clip = imgs.permute(0, 2, 1, 3, 4)   # [B, C, T, H, W]
                    h_target = _layerlist_to_h(target_encoder(full_clip))
                    # h_target: [B, T_tok*HW, n_layers*D]  — ground truth for loss

                # ------------------------------------------------------------ #
                # 2. CONTEXT: context encoder sees only the first n_ctx_tubelets
                # ------------------------------------------------------------ #
                ctx_frames = n_ctx_tubelets * tubelet_size    # raw frames for context
                ctx_clip = imgs[:, :ctx_frames, :, :, :].permute(0, 2, 1, 3, 4)
                # encoder runs WITH grad so it can be fine-tuned (enc_lr_scale)
                ctx_layer_outs = encoder(ctx_clip)            # list[Tensor[B, n_ctx*HW, D]]
                h_context = _layerlist_to_h(ctx_layer_outs)  # [B, n_ctx*HW, n_layers*D]

                # ------------------------------------------------------------ #
                # 3. PAD future frames with zeros → predictor input
                # The causal mask prevents future tokens from attending to each
                # other, so zero-padding is safe: they carry no signal.
                # ------------------------------------------------------------ #
                B_sz, N_ctx, D_full = h_context.shape
                N_future = (T_tok - n_ctx_tubelets) * HW
                padding = torch.zeros(
                    B_sz, N_future, D_full,
                    device=h_context.device, dtype=h_context.dtype,
                )
                h_predictor_input = torch.cat([h_context, padding], dim=1)
                # h_predictor_input: [B, T_tok*HW, n_layers*D]

                # ------------------------------------------------------------ #
                # 4. PREDICTOR: roll out all T_tok tubelets from actions
                # ------------------------------------------------------------ #
                def forward_predictor():
                    preds, ctx_preds = predictor(
                        h_predictor_input,
                        actions,
                        states,
                        intrinsics=intrinsics if use_intrinsics else None,
                    )
                    return preds, ctx_preds

                # ------------------------------------------------------------ #
                # 5. LOSS: predictor output vs frozen target features
                # ------------------------------------------------------------ #
                def loss_fn(preds, ctx_preds, h_tgt):
                    embed_dim = h_tgt.shape[-1] // n_hierarchical_layers
                    pred_chunks = preds.split(embed_dim, dim=-1)
                    h_chunks    = h_tgt.split(embed_dim, dim=-1)
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

                    loss_ctx = preds.new_zeros(())
                    if ctx_preds is not None:
                        # Dense context loss: only compare the context tubelet slots
                        n_ctx_toks = n_ctx_tubelets * HW
                        ctx_chunks = ctx_preds[:, :n_ctx_toks].split(embed_dim, dim=-1)
                        h_ctx_chunks = h_tgt[:, :n_ctx_toks].split(embed_dim, dim=-1)
                        for p, t in zip(ctx_chunks, h_ctx_chunks):
                            t = t.detach()
                            if normalize_reps:
                                p = F.layer_norm(p, (p.size(-1),))
                                t = F.layer_norm(t, (t.size(-1),))
                            loss_ctx = loss_ctx + (
                                torch.mean(torch.abs(p - t) ** loss_exp) / loss_exp
                            )
                        loss_ctx = loss_ctx / n_hierarchical_layers

                    return loss_pred + loss_ctx, loss_pred, loss_ctx

                with torch.amp.autocast("cuda", dtype=dtype, enabled=mixed_precision):
                    preds, ctx_preds = forward_predictor()
                    loss, loss_pred, loss_ctx = loss_fn(preds, ctx_preds, h_target)

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

                return loss.item(), loss_pred.item(), loss_ctx.item(), _new_lr, _new_wd

            (loss, loss_pred, loss_ctx, _new_lr, _new_wd), gpu_etime_ms = gpu_timer(train_step)
            iter_elapsed_time_ms = (time.time() - itr_start_time) * 1000.0

            loss_meter.update(loss)
            loss_pred_meter.update(loss_pred)
            loss_ctx_meter.update(loss_ctx)
            iter_time_meter.update(iter_elapsed_time_ms)
            gpu_time_meter.update(gpu_etime_ms)
            data_elapsed_time_meter.update(data_elapsed_time_ms)

            def log_stats():
                csv_logger.log(
                    epoch + 1, itr, loss, loss_pred, loss_ctx,
                    iter_elapsed_time_ms, gpu_etime_ms, data_elapsed_time_ms,
                )
                if (itr % log_freq == 0) or (itr == ipe - 1) or np.isnan(loss) or np.isinf(loss):
                    logger.info(
                        "[%d, %5d] loss: %.3f [pred: %.3f ctx: %.3f] "
                        "[wd: %.2e] [lr: %.2e] "
                        "[mem: %.2e] "
                        "[iter: %.1f ms] [gpu: %.1f ms] [data: %.1f ms]"
                        % (
                            epoch + 1, itr,
                            loss_meter.avg, loss_pred_meter.avg, loss_ctx_meter.avg,
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

        # -- accumulate epoch-level stats for curves
        training_stats["loss"].append(loss_meter.avg)
        training_stats["loss_pred"].append(loss_pred_meter.avg)
        training_stats["loss_ctx"].append(loss_ctx_meter.avg)
        training_stats["lr"].append(scheduler.get_last_lr() if hasattr(scheduler, "get_last_lr") else _new_lr)
        training_stats["wd"].append(_new_wd)
        training_stats["iter_ms"].append(iter_time_meter.avg)
        training_stats["gpu_ms"].append(gpu_time_meter.avg)
        training_stats["mem_gb"].append(torch.cuda.max_memory_allocated() / 1024.0**3)

        if epoch % CHECKPOINT_FREQ == 0 or epoch == (num_epochs - 1):
            save_checkpoint(epoch + 1, latest_path)
            if save_every_freq > 0 and epoch % save_every_freq == 0:
                save_every_path = os.path.join(scratch_folder, f"e{epoch}.pt")
                save_checkpoint(epoch + 1, save_every_path)
                prev_save = os.path.join(scratch_folder, f"e{epoch - save_every_freq}.pt")
                if rank == 0 and os.path.exists(prev_save):
                    os.remove(prev_save)
                    logger.info(f"Removed old checkpoint: {prev_save}")

                # -- training curves
                if rank == 0:
                    from pathlib import Path
                    try:
                        plot_training_curves(
                            training_stats,
                            Path(folder) / f"training_curves_e{epoch:03d}.png",
                        )
                        logger.info(f"Training curves saved to {folder}/training_curves_e{epoch:03d}.png")
                    except Exception as _ce:
                        logger.warning(f"Training curves failed: {_ce}")

        # -- PCA feature visualisation on test scenes
        if pca_loader is not None and (epoch % VIZ_FREQ == 0 or epoch == (num_epochs - 1)):
            logger.info("Running PCA feature visualisation...")
            grid_h = grid_w = crop_size // patch_size
            HW = grid_h * grid_w
            T_tok = seq_len // tubelet_size
            ctx_frames_eval = n_ctx_tubelets * tubelet_size
            _enc = target_encoder.module if hasattr(target_encoder, "module") else target_encoder
            _pred = predictor.module if hasattr(predictor, "module") else predictor
            _enc.eval()
            _pred.eval()
            with torch.no_grad():
                for vis_idx, vis_sample in enumerate(pca_loader):
                    if vis_idx >= n_pca_scenes:
                        break
                    v_imgs  = vis_sample["images"].to(device)
                    v_states     = vis_sample["states"].to(device, dtype=torch.float)[:, ::tubelet_size]
                    v_actions    = vis_sample["actions"].to(device, dtype=torch.float)[:, ::tubelet_size]
                    v_intrinsics = vis_sample["intrinsics"].to(device, dtype=torch.float)[:, ::tubelet_size]
                    full_clip = v_imgs.permute(0, 2, 1, 3, 4)
                    layer_outs = _enc(full_clip)
                    if normalize_reps:
                        layer_outs = [F.layer_norm(f, (f.size(-1),)) for f in layer_outs]
                    h_tgt = torch.cat(layer_outs, dim=-1)[0]   # [T*HW, L*D]
                    _layer_D = layer_outs[-1].shape[-1]
                    h_tgt_last = h_tgt[..., -_layer_D:]        # [T*HW, D] last layer only
                    ctx_clip = v_imgs[:, :ctx_frames_eval].permute(0, 2, 1, 3, 4)
                    ctx_lo = _enc(ctx_clip)
                    if normalize_reps:
                        ctx_lo = [F.layer_norm(f, (f.size(-1),)) for f in ctx_lo]
                    h_ctx = torch.cat(ctx_lo, dim=-1)           # [B, n_ctx*HW, L*D]
                    B_v, N_ctx_v, D_v = h_ctx.shape
                    pad_v = torch.zeros(B_v, T_tok * HW - N_ctx_v, D_v, device=device, dtype=h_ctx.dtype)
                    h_in  = torch.cat([h_ctx, pad_v], dim=1)
                    preds_v, _ = _pred(
                        h_in, v_actions, v_states,
                        intrinsics=v_intrinsics if use_intrinsics else None,
                    )
                    h_pred_last = preds_v[0][..., -_layer_D:]  # [T*HW, D] last layer only
                    out_path = os.path.join(pca_vis_dir, f"e{epoch:03d}_scene{vis_idx:02d}.png")
                    try:
                        visualize_pca_features(
                            h_gt=h_tgt_last, h_pred=h_pred_last,
                            imgs=v_imgs[0], grid_h=grid_h, grid_w=grid_w,
                            out_path=out_path, n_frames=4,
                        )
                    except Exception as _ve:
                        logger.warning(f"PCA vis failed for scene {vis_idx}: {_ve}")
            _enc.train()
            _pred.train()
            logger.info(f"PCA visualisations saved to {pca_vis_dir}")
