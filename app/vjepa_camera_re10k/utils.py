# Copyright (c) Meta Platforms, Inc. and affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

import logging
import os
import sys

import torch

import src.models.camera_ac_predictor as cam_pred
# V-JEPA 2.1 vision transformer (image path). The V-JEPA 2 transformer
# (src.models.vision_transformer) has no img_temporal_dim_size / patch_embed_img
# and is video-only with tubelet_size=2. CameraAC now consumes single images
# via V-JEPA 2.1's patch_embed_img; see _make_vjepa2_1_model in src/hub/backbones.py.
import app.vjepa_2_1.models.vision_transformer as video_vit
from src.utils.checkpoint_loader import robust_checkpoint_loader
from src.utils.schedulers import CosineWDSchedule, WSDSchedule

logging.basicConfig(stream=sys.stdout, level=logging.INFO)
logger = logging.getLogger()


def _env_bool(name, default):
    """Parse a boolean environment variable. Only override when explicitly set."""
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    return raw.strip().lower() in ("1", "true", "yes", "on")


def _env_int(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return int(raw)
    except ValueError:
        return default


def _env_float(name, default):
    raw = os.environ.get(name)
    if raw is None or raw == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int_list(name):
    """Parse a comma-separated integer list from an env var.

    Returns ``None`` when the variable is unset / empty / invalid, signalling
    the caller should fall back to the default layer-selection policy.
    """
    raw = os.environ.get(name)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        values = [int(part.strip()) for part in str(raw).split(",") if part.strip()]
    except ValueError:
        logger.warning(f"Ignoring malformed {name}={raw!r}; expected comma-separated ints")
        return None
    return values or None


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
        # Fail loud when the live predictor is missing experiment-flag submodules
        # present in the checkpoint. Silent drops invalidate ablation metrics.
        if key == "predictor":
            unexpected = list(getattr(msg, "unexpected_keys", []) or [])
            flag_prefixes = (
                "ray_pe_mlp.",
                "predictor_proj_delta.",
                "residual_refine.",
                "residual_gate",
            )
            leaked = [k for k in unexpected if any(k.startswith(p) for p in flag_prefixes)]
            if leaked:
                raise RuntimeError(
                    "Predictor checkpoint contains experiment-flag weights that "
                    "are not present in the live model: "
                    f"{leaked}. The predictor was built without the matching "
                    "USE_RAY_PE / USE_DELTA_HEAD / USE_RESIDUAL_HEAD flags. "
                    "Set the corresponding environment variable (or kwarg) so "
                    "init_video_model instantiates the right modules before "
                    "loading this checkpoint."
                )

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


def encode_clip_as_images(encoder, clip_bcthw):
    """Run a V-JEPA 2.1 ViT (with img_temporal_dim_size=1) via the image path.

    The V-JEPA 2.1 ViT auto-selects between patch_embed_img (image path,
    used when x.shape[2] == img_temporal_dim_size == 1) and patch_embed
    (video path, used when tubeleting T frames together). CameraAC now
    encodes each frame as a single image: T is flattened into the batch
    dimension so each forward sees (B*T, C, 1, H, W) and takes the image path.

    Args:
        encoder: a V-JEPA 2.1 VisionTransformer built with img_temporal_dim_size=1
            and out_layers set (so forward returns a list of per-layer tensors).
        clip_bcthw: (B, C, T, H, W) clip tensor.

    Returns:
        A list of per-layer tensors, each shape (B, T*HW, D) where HW is the
        per-image token count (e.g. 576 for img_size=384, patch_size=16).
        The (T*HW) layout matches the concatenation CameraAC's train.py
        expects: per-tubelet slices of size HW laid out along the token axis.
    """
    if clip_bcthw.ndim != 5:
        raise ValueError(
            f"encode_clip_as_images expects (B,C,T,H,W); got shape={tuple(clip_bcthw.shape)}"
        )
    B, C, T, H, W = clip_bcthw.shape
    # (B, C, T, H, W) -> (B, T, C, H, W) -> (B*T, C, H, W) -> (B*T, C, 1, H, W)
    imgs = clip_bcthw.permute(0, 2, 1, 3, 4).reshape(B * T, C, H, W).unsqueeze(2)
    layer_outs = encoder(imgs)
    if not isinstance(layer_outs, (list, tuple)):
        raise RuntimeError(
            "V-JEPA 2.1 encoder returned a single tensor; CameraAC requires out_layers "
            "to be set so the forward returns a list of per-layer tensors."
        )
    reshaped = []
    for lo in layer_outs:
        if lo.ndim != 3:
            raise RuntimeError(
                f"Expected per-layer output shape (B*T, HW, D); got {tuple(lo.shape)}"
            )
        BT_, HW_, D_ = lo.shape
        if BT_ != B * T:
            raise RuntimeError(
                f"V-JEPA 2.1 image-path batch mismatch: expected {B*T} got {BT_}"
            )
        # (B*T, HW, D) -> (B, T, HW, D) -> (B, T*HW, D)
        reshaped.append(lo.reshape(B, T, HW_, D_).reshape(B, T * HW_, D_))
    return reshaped


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
    use_ray_pe=False,
    ray_pe_dim=6,
    ray_pe_hidden=256,
    use_delta_head=False,
    use_residual_head=False,
    residual_head_depth=2,
    residual_head_ratio=2.0,
):
    # Ablation-flag env-var fallback. Callers that do not explicitly pass the
    # experiment flags (e.g. the rollout eval tool) still build the correct
    # predictor when USE_RAY_PE / USE_DELTA_HEAD / USE_RESIDUAL_HEAD are set
    # by the orchestration shell script. Explicit True kwargs are never demoted.
    use_ray_pe = bool(use_ray_pe) or _env_bool("USE_RAY_PE", False)
    ray_pe_dim = _env_int("RAY_PE_DIM", ray_pe_dim)
    ray_pe_hidden = _env_int("RAY_PE_HIDDEN", ray_pe_hidden)
    use_delta_head = bool(use_delta_head) or _env_bool("USE_DELTA_HEAD", False)
    use_residual_head = bool(use_residual_head) or _env_bool("USE_RESIDUAL_HEAD", False)
    residual_head_depth = _env_int("RESIDUAL_HEAD_DEPTH", residual_head_depth)
    residual_head_ratio = _env_float("RESIDUAL_HEAD_RATIO", residual_head_ratio)
    logger.info(
        "init_video_model experiment flags: "
        f"use_ray_pe={use_ray_pe} (dim={ray_pe_dim}, hidden={ray_pe_hidden}) "
        f"use_delta_head={use_delta_head} "
        f"use_residual_head={use_residual_head} (depth={residual_head_depth}, ratio={residual_head_ratio})"
    )

    # Build the encoder. Instantiate once without out_layers so we can read the
    # canonical V-JEPA 2.1 hierarchical_layers (which the ViT hardcodes per
    # depth: [5,11,17,23] for depth 24, [11,23,37,47] for depth 48, etc.).
    # V-JEPA 2.1's forward indexes norms_block via hierarchical_layers.index(i),
    # so out_layers MUST be a subset of hierarchical_layers.
    #
    # img_temporal_dim_size=1 activates patch_embed_img (single-image path) in
    # addition to patch_embed (video path); the forward auto-selects based on
    # the input's T dimension (==1 -> image path). interpolate_rope=True matches
    # the _make_vjepa2_1_model recipe in src/hub/backbones.py.
    _vit_kwargs = dict(
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
        img_temporal_dim_size=1,
        interpolate_rope=True,
    )
    _enc_tmp = video_vit.__dict__[model_name](**_vit_kwargs)
    n_blocks = len(_enc_tmp.blocks)
    canonical_hier = list(_enc_tmp.hierarchical_layers)
    del _enc_tmp
    # Layer-selection policy.
    #   * Default: the canonical V-JEPA 2.1 hierarchical layers
    #     (e.g. [5,11,17,23] for vit_large). These are the only indices with
    #     pretrained norms_block weights in the V-JEPA 2.1 checkpoint.
    #   * Override: set VJEPA_OUT_LAYERS to an explicit list that is a
    #     subset of the canonical hierarchical layers. Arbitrary (non-canonical)
    #     indices are rejected because they would index norms_block out of range.
    out_layers_override = _env_int_list("VJEPA_OUT_LAYERS")
    if out_layers_override is not None:
        if len(out_layers_override) != n_hierarchical_layers:
            raise ValueError(
                f"VJEPA_OUT_LAYERS has {len(out_layers_override)} entries "
                f"({out_layers_override}) but n_hierarchical_layers={n_hierarchical_layers}. "
                f"They must match because the predictor splits target features by this count."
            )
        for idx in out_layers_override:
            if idx not in canonical_hier:
                raise ValueError(
                    f"VJEPA_OUT_LAYERS index {idx} is not in V-JEPA 2.1 canonical "
                    f"hierarchical_layers {canonical_hier} for an encoder with {n_blocks} "
                    f"blocks. The V-JEPA 2.1 ViT only has norms_block weights for these "
                    f"indices, so picking others would break checkpoint loading and forward."
                )
        out_layers = list(out_layers_override)
        logger.info(f"Using VJEPA_OUT_LAYERS override: {out_layers} (canonical={canonical_hier})")
    else:
        if n_hierarchical_layers != len(canonical_hier):
            raise ValueError(
                f"n_hierarchical_layers={n_hierarchical_layers} does not match the V-JEPA 2.1 "
                f"canonical hierarchical_layers={canonical_hier} (len={len(canonical_hier)}). "
                f"Set model.n_hierarchical_layers={len(canonical_hier)} in the YAML or "
                f"override with VJEPA_OUT_LAYERS."
            )
        out_layers = list(canonical_hier)
        logger.info(
            f"Using canonical V-JEPA 2.1 hierarchical out_layers: {out_layers} "
            f"(n_blocks={n_blocks})"
        )
    encoder = video_vit.__dict__[model_name](
        **_vit_kwargs,
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
        use_ray_pe=use_ray_pe,
        ray_pe_dim=ray_pe_dim,
        ray_pe_hidden=ray_pe_hidden,
        use_delta_head=use_delta_head,
        use_residual_head=use_residual_head,
        residual_head_depth=residual_head_depth,
        residual_head_ratio=residual_head_ratio,
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
