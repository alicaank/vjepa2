# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import sys

import torch

import src.models.camera_ac_predictor as cam_pred
import src.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _upgrade_camera_predictor_state_dict(pretrained_dict, model):
    model_keys = set(model.state_dict().keys())

    def _upgrade_projection(prefix):
        old_weight_key = f"{prefix}.weight"
        old_bias_key = f"{prefix}.bias"
        first_head_weight_key = f"{prefix}.0.weight"
        first_head_bias_key = f"{prefix}.0.bias"
        if first_head_weight_key not in model_keys or old_weight_key not in pretrained_dict:
            return
        head_weight_shape = model.state_dict()[first_head_weight_key].shape
        old_weight = pretrained_dict.pop(old_weight_key)
        if old_weight.shape[0] % head_weight_shape[0] != 0:
            pretrained_dict[old_weight_key] = old_weight
            return
        n_heads = old_weight.shape[0] // head_weight_shape[0]
        for head_idx in range(n_heads):
            start = head_idx * head_weight_shape[0]
            end = (head_idx + 1) * head_weight_shape[0]
            pretrained_dict[f"{prefix}.{head_idx}.weight"] = old_weight[start:end]
        if old_bias_key in pretrained_dict and first_head_bias_key in model_keys:
            old_bias = pretrained_dict.pop(old_bias_key)
            head_bias_shape = model.state_dict()[first_head_bias_key].shape
            if old_bias.shape[0] % head_bias_shape[0] != 0:
                pretrained_dict[old_bias_key] = old_bias
                return
            n_bias_heads = old_bias.shape[0] // head_bias_shape[0]
            for head_idx in range(n_bias_heads):
                start = head_idx * head_bias_shape[0]
                end = (head_idx + 1) * head_bias_shape[0]
                pretrained_dict[f"{prefix}.{head_idx}.bias"] = old_bias[start:end]

    for prefix in ("predictor_proj", "module.predictor_proj", "predictor_proj_context", "module.predictor_proj_context"):
        _upgrade_projection(prefix)
    return pretrained_dict


def load_pretrained(
    r_path,
    encoder=None,
    predictor=None,
    target_encoder=None,
    context_encoder_key="encoder",
    target_encoder_key="target_encoder",
    load_predictor=False,
    load_encoder=True,
):
    logger.info(f"Loading pretrained model from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]

    if load_encoder:
        pretrained_dict = checkpoint[context_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")

    if load_predictor:
        pretrained_dict = checkpoint["predictor"]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = _upgrade_camera_predictor_state_dict(pretrained_dict, predictor)
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")

    if load_encoder and target_encoder is not None:
        pretrained_dict = checkpoint[target_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")

    del checkpoint
    return encoder, predictor, target_encoder


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt=None,
    scaler=None,
    replace_kw=["backbone."],
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    checkpoint_type = checkpoint.get("checkpoint_type", "full")
    logger.info(f"checkpoint type: {checkpoint_type}")

    if "target_encoder" not in checkpoint and "encoder" in checkpoint:
        checkpoint["target_encoder"] = checkpoint["encoder"]

    for key, model in [("encoder", encoder), ("predictor", predictor), ("target_encoder", target_encoder)]:
        if model is None:
            continue
        if key not in checkpoint:
            logger.warning(f"Checkpoint is missing {key}; skipping load for that module.")
            continue
        pretrained_dict = checkpoint[key]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        if key == "predictor":
            pretrained_dict = _upgrade_camera_predictor_state_dict(pretrained_dict, model)
        msg = model.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded {key} from epoch {epoch} with msg: {msg}")

    if opt is not None:
        opt_state = checkpoint.get("opt")
        if opt_state is not None:
            opt.load_state_dict(opt_state)
        else:
            logger.warning("Checkpoint has no optimizer state; resuming from weights-only checkpoint.")
    if scaler is not None:
        scaler_state = checkpoint.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        else:
            logger.warning("Checkpoint has no scaler state; resuming from weights-only checkpoint.")

    logger.info(f"read-path: {r_path}")
    del checkpoint
    return encoder, predictor, target_encoder, opt, scaler, epoch


def init_video_model(
    device,
    patch_size=16,
    max_num_frames=8,
    tubelet_size=2,
    model_name="vit_large",
    crop_size=256,
    pred_depth=12,
    pred_num_heads=None,
    pred_embed_dim=1024,
    n_hierarchical_layers=4,
    out_embed_dim=None,
    uniform_power=True,
    use_sdpa=True,
    use_rope=True,
    use_silu=False,
    use_pred_silu=False,
    wide_silu=True,
    pred_is_frame_causal=True,
    use_activation_checkpointing=False,
    state_dim=7,
    action_dim=7,
    intrinsics_dim=4,
    use_intrinsics=True,
    predict_all=True,
):
    # Build the encoder.  We instantiate it first without out_layers to read
    # the total block depth, then reinitialise with the last n_hierarchical_layers.
    _enc_tmp = video_vit.__dict__[model_name](
        img_size=crop_size, patch_size=patch_size, num_frames=max_num_frames,
        tubelet_size=tubelet_size, uniform_power=uniform_power, use_sdpa=use_sdpa,
        use_silu=use_silu, wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing, use_rope=use_rope,
    )
    n_blocks = len(_enc_tmp.blocks)
    del _enc_tmp
    # Collect the last n_hierarchical_layers block indices
    out_layers = list(range(n_blocks - n_hierarchical_layers, n_blocks))
    encoder = video_vit.__dict__[model_name](
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        uniform_power=uniform_power,
        use_sdpa=use_sdpa,
        use_silu=use_silu,
        wide_silu=wide_silu,
        use_activation_checkpointing=use_activation_checkpointing,
        use_rope=use_rope,
        out_layers=out_layers,
    )

    _out_embed_dim = out_embed_dim if out_embed_dim is not None else encoder.embed_dim

    predictor = cam_pred.vit_camera_ac_predictor(
        img_size=crop_size,
        patch_size=patch_size,
        num_frames=max_num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        n_hierarchical_layers=n_hierarchical_layers,
        out_embed_dim=_out_embed_dim,
        depth=pred_depth,
        num_heads=encoder.num_heads if pred_num_heads is None else pred_num_heads,
        uniform_power=uniform_power,
        use_rope=use_rope,
        use_silu=use_pred_silu,
        wide_silu=wide_silu,
        is_frame_causal=pred_is_frame_causal,
        use_activation_checkpointing=use_activation_checkpointing,
        state_dim=state_dim,
        action_dim=action_dim,
        intrinsics_dim=intrinsics_dim,
        use_intrinsics=use_intrinsics,
        predict_all=predict_all,
    )

    encoder.to(device)
    predictor.to(device)
    logger.info(encoder)
    logger.info(predictor)

    def count_parameters(model):
        return sum(p.numel() for p in model.parameters() if p.requires_grad)

    logger.info(f"Encoder number of parameters: {count_parameters(encoder)}")
    logger.info(f"Predictor number of parameters: {count_parameters(predictor)}")

    return encoder, predictor


def init_opt(
    encoder,
    predictor,
    iterations_per_epoch,
    start_lr,
    ref_lr,
    warmup,
    anneal,
    num_epochs,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    param_groups = [
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" not in n) and (len(p.shape) != 1)),
        },
        {
            "params": (p for n, p in encoder.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        },
        {
            "params": (p for n, p in predictor.named_parameters() if ("bias" in n) or (len(p.shape) == 1)),
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        },
    ]

    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    scheduler = WSDSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        anneal_steps=int(anneal * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    wd_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=wd,
        final_wd=final_wd,
        T_max=int(num_epochs * iterations_per_epoch),
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
    return optimizer, scaler, scheduler, wd_scheduler
