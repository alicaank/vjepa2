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


def _lookup_state_dict_tensor(state_dict, key):
    if key in state_dict:
        return key, state_dict[key]
    if key.startswith("module."):
        bare_key = key[len("module."):]
        if bare_key in state_dict:
            return bare_key, state_dict[bare_key]
    else:
        module_key = f"module.{key}"
        if module_key in state_dict:
            return module_key, state_dict[module_key]
    return None, None


def _normalize_state_dict_module_prefix(pretrained_dict, model):
    """Match a checkpoint state-dict's DDP ``module.`` prefix to ``model``.

    ``load_state_dict(..., strict=False)`` is dangerously quiet when every key
    misses due only to DDP wrapping. Normalize the whole checkpoint prefix before
    shape adaptation so frozen unwrapped encoders and DDP-wrapped target
    encoders both receive the same pretrained weights.
    """
    if not pretrained_dict or model is None:
        return pretrained_dict
    live_keys = set(model.state_dict().keys())
    if not live_keys:
        return pretrained_dict
    ckpt_keys = list(pretrained_dict.keys())
    ckpt_is_wrapped = all(k.startswith("module.") for k in ckpt_keys)
    live_is_wrapped = any(k.startswith("module.") for k in live_keys)
    if ckpt_is_wrapped and not live_is_wrapped:
        return {k[len("module."):]: v for k, v in pretrained_dict.items()}
    if live_is_wrapped and not ckpt_is_wrapped:
        return {f"module.{k}": v for k, v in pretrained_dict.items()}
    return pretrained_dict


def _raise_if_zero_key_load(load_msg, model, *, module_name, loader_name):
    live_keys = set(model.state_dict().keys()) if model is not None else set()
    n_expected = len(live_keys)
    n_missing = len(getattr(load_msg, "missing_keys", []) or [])
    if n_expected > 0 and n_missing >= n_expected:
        raise RuntimeError(
            f"{loader_name}: {module_name} matched 0 keys "
            f"(missing={n_missing}/{n_expected}). Checkpoint keys likely "
            "have a different prefix (for example DDP ``module.``) than the "
            "live model. Aborting so training/eval does not run with a "
            "randomly initialized module."
        )


def _upgrade_camera_encoder_state_dict(pretrained_dict, model):
    """Adapt V-JEPA 2.1 encoder weights for the CameraAC image-path setup.

    CameraAC encodes each frame independently through ``patch_embed_img``
    (temporal kernel = 1). The downloaded V-JEPA 2.1 checkpoint still carries
    the video-path ``patch_embed`` kernel for the backbone's native
    ``tubelet_size`` (e.g. 2), and PyTorch raises even with ``strict=False``
    when that unused Conv3d weight has a different shape. Prefer the checkpoint's
    image-path kernel when available; otherwise collapse the temporal dimension.
    """
    upgraded = dict(pretrained_dict)
    model_state = model.state_dict()
    remapped = []
    dropped = []

    for model_key, model_tensor in model_state.items():
        ckpt_key, ckpt_tensor = _lookup_state_dict_tensor(upgraded, model_key)
        if ckpt_tensor is None or ckpt_tensor.shape == model_tensor.shape:
            continue

        if model_key.endswith("patch_embed.proj.weight"):
            img_key = model_key.replace("patch_embed.proj.weight", "patch_embed_img.proj.weight")
            _, img_tensor = _lookup_state_dict_tensor(upgraded, img_key)
            if img_tensor is not None and img_tensor.shape == model_tensor.shape:
                upgraded[model_key] = img_tensor.detach().clone()
                remapped.append(
                    f"{model_key} <- {img_key} {tuple(img_tensor.shape)}"
                )
                continue

            if (
                ckpt_tensor.ndim == model_tensor.ndim == 5
                and ckpt_tensor.shape[:2] == model_tensor.shape[:2]
                and ckpt_tensor.shape[3:] == model_tensor.shape[3:]
                and model_tensor.shape[2] == 1
            ):
                upgraded[model_key] = ckpt_tensor.mean(dim=2, keepdim=True)
                remapped.append(
                    f"{model_key} <- temporal-mean({tuple(ckpt_tensor.shape)} -> {tuple(model_tensor.shape)})"
                )
                continue

        upgraded.pop(ckpt_key, None)
        dropped.append(
            f"{ckpt_key} checkpoint={tuple(ckpt_tensor.shape)} model={tuple(model_tensor.shape)}"
        )

    if remapped:
        logger.info(
            "Adapted pretrained encoder weights for CameraAC image-path loading: %s",
            "; ".join(remapped),
        )
    if dropped:
        logger.warning(
            "Dropped incompatible pretrained encoder tensors during CameraAC load: %s",
            "; ".join(dropped),
        )

    return upgraded


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
        pretrained_dict = _normalize_state_dict_module_prefix(pretrained_dict, encoder)
        pretrained_dict = _upgrade_camera_encoder_state_dict(pretrained_dict, encoder)
        msg = encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained encoder from epoch {epoch} with msg: {msg}")
        _raise_if_zero_key_load(
            msg,
            encoder,
            module_name="encoder",
            loader_name="load_pretrained",
        )

    if load_predictor:
        pretrained_dict = checkpoint["predictor"]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = _normalize_state_dict_module_prefix(pretrained_dict, predictor)
        pretrained_dict = _upgrade_camera_predictor_state_dict(pretrained_dict, predictor)
        msg = predictor.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained predictor from epoch {epoch} with msg: {msg}")
        _raise_if_zero_key_load(
            msg,
            predictor,
            module_name="predictor",
            loader_name="load_pretrained",
        )

    if load_encoder and target_encoder is not None:
        pretrained_dict = checkpoint[target_encoder_key]
        pretrained_dict = {k.replace("backbone.", ""): v for k, v in pretrained_dict.items()}
        pretrained_dict = _normalize_state_dict_module_prefix(pretrained_dict, target_encoder)
        pretrained_dict = _upgrade_camera_encoder_state_dict(pretrained_dict, target_encoder)
        msg = target_encoder.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded pretrained target encoder from epoch {epoch} with msg: {msg}")
        _raise_if_zero_key_load(
            msg,
            target_encoder,
            module_name="target_encoder",
            loader_name="load_pretrained",
        )

    del checkpoint
    return encoder, predictor, target_encoder


def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt=None,
    scaler=None,
    corrector=None,
    latent_patchgan=None,
    latent_patchgan_opt=None,
    replace_kw=["backbone."],
):
    logger.info(f"Loading checkpoint from {r_path}")
    checkpoint = robust_checkpoint_loader(r_path, map_location=torch.device("cpu"))
    epoch = checkpoint["epoch"]
    checkpoint_type = checkpoint.get("checkpoint_type", "full")
    logger.info(f"checkpoint type: {checkpoint_type}")

    if "target_encoder" not in checkpoint and "encoder" in checkpoint:
        checkpoint["target_encoder"] = checkpoint["encoder"]

    model_key_pairs = [("encoder", encoder), ("predictor", predictor), ("target_encoder", target_encoder)]
    if corrector is not None:
        # Corrector load is best-effort: pre-corrector checkpoints simply
        # don't have the key, in which case the corrector keeps its small
        # ~zero init. This matches the design doc's backward-compat rule
        # (§5.6, ckpt key remap).
        if "corrector" in checkpoint:
            model_key_pairs.append(("corrector", corrector))
        else:
            logger.warning(
                "Resume checkpoint has no 'corrector' key; corrector retains "
                "its initialization (residual_scale_init~=0, near no-op)."
            )
    if latent_patchgan is not None:
        if "latent_patchgan" in checkpoint:
            model_key_pairs.append(("latent_patchgan", latent_patchgan))
        else:
            logger.warning(
                "Resume checkpoint has no 'latent_patchgan' key; discriminator "
                "retains its initialization."
            )

    for key, model in model_key_pairs:
        if model is None:
            continue
        if key not in checkpoint:
            logger.warning(f"Checkpoint is missing {key}; skipping load for that module.")
            continue
        pretrained_dict = checkpoint[key]
        for kw in replace_kw:
            pretrained_dict = {k.replace(kw, ""): v for k, v in pretrained_dict.items()}
        live_keys = set(model.state_dict().keys())
        pretrained_dict = _normalize_state_dict_module_prefix(pretrained_dict, model)
        if key in ("encoder", "target_encoder"):
            pretrained_dict = _upgrade_camera_encoder_state_dict(pretrained_dict, model)
        if key == "predictor":
            pretrained_dict = _upgrade_camera_predictor_state_dict(pretrained_dict, model)
        msg = model.load_state_dict(pretrained_dict, strict=False)
        logger.info(f"loaded {key} from epoch {epoch} with msg: {msg}")
        # Fail loud when a non-trivial load matched zero keys — this almost
        # always means a prefix / naming mismatch silently zeroed the model.
        _raise_if_zero_key_load(
            msg,
            model,
            module_name=key,
            loader_name="load_checkpoint",
        )
        # Fail loud when the live predictor is missing experiment-flag submodules
        # present in the checkpoint. Silent drops invalidate ablation metrics.
        if key == "predictor":
            unexpected = list(getattr(msg, "unexpected_keys", []) or [])
            flag_prefixes = (
                "ray_pe_mlp.",
                "predictor_proj_delta.",
                "residual_refine.",
                "residual_gate",
                "completion_refine.",
                "completion_gate",
                "canvas_beta_by_type",
                "canvas_mask_proj.",
                "canvas_type_embed.",
                "canvas_target_step_embed.",
                "camera_ucpe_branches.",
                "predictor_adaln_conditioner.",
                "predictor_adaln_modulators.",
            )
            leaked = [k for k in unexpected if any(k.startswith(p) for p in flag_prefixes)]
            if leaked:
                raise RuntimeError(
                    "Predictor checkpoint contains experiment-flag weights that "
                    "are not present in the live model: "
                    f"{leaked}. The predictor was built without the matching "
                    "USE_RAY_PE / USE_DELTA_HEAD / USE_RESIDUAL_HEAD / "
                    "TARGET_SLOT_MODE / CAMERA_UCPE_ENABLED flags. "
                    "Set the corresponding environment variable (or kwarg) so "
                    "init_video_model instantiates the right modules before "
                    "loading this checkpoint."
                )

    if opt is not None:
        opt_state = checkpoint.get("opt")
        if opt_state is not None:
            try:
                opt.load_state_dict(opt_state)
            except ValueError as e:
                logger.warning(f"Skipping optimizer state load because it is incompatible with the live model: {e}")
        else:
            logger.warning("Checkpoint has no optimizer state; resuming from weights-only checkpoint.")
    if scaler is not None:
        scaler_state = checkpoint.get("scaler")
        if scaler_state is not None:
            scaler.load_state_dict(scaler_state)
        else:
            logger.warning("Checkpoint has no scaler state; resuming from weights-only checkpoint.")
    if latent_patchgan_opt is not None:
        patchgan_opt_state = checkpoint.get("latent_patchgan_opt")
        if patchgan_opt_state is not None:
            try:
                latent_patchgan_opt.load_state_dict(patchgan_opt_state)
            except ValueError as e:
                logger.warning(
                    "Skipping latent PatchGAN optimizer state load because it "
                    f"is incompatible with the live discriminator: {e}"
                )
        else:
            logger.warning(
                "Checkpoint has no latent_patchgan_opt state; discriminator "
                "optimizer starts fresh."
            )

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
    use_four_layer_birth_head=False,
    completion_head_hidden=512,
    use_fsq_head=False,
    fsq_total_code_axes=0,
    fsq_levels=0,
    warp_context_latents=False,
    warp_context_padding_mode="border",
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
    predictor_adaln_enabled=False,
    predictor_adaln_hidden=512,
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
    encoder_backbone="vjepa21",
    encoder_freeze=False,
):
    # Encoder backbone selection. ``vjepa21`` (default) preserves the
    # canonical V-JEPA 2.1 behaviour bit-for-bit. ``mast3r`` and
    # ``cradio`` swap in frozen feature adapters with the same forward
    # contract (list of ``(B, N, D)`` feature tensors).
    encoder_backbone = os.environ.get("ENCODER_BACKBONE", encoder_backbone) or "vjepa21"
    encoder_backbone = str(encoder_backbone).lower().strip()
    if encoder_backbone not in ("vjepa21", "mast3r", "cradio"):
        raise ValueError(
            f"encoder_backbone={encoder_backbone!r} unsupported. "
            f"Expected 'vjepa21', 'mast3r', or 'cradio'."
        )
    if os.environ.get("ENCODER_FREEZE") is not None:
        encoder_freeze = _env_bool("ENCODER_FREEZE", encoder_freeze)
    encoder_freeze = bool(encoder_freeze)

    # Ablation-flag env-var fallback. Callers that do not explicitly pass the
    # experiment flags (e.g. the rollout eval tool) still build the correct
    # predictor when USE_RAY_PE / USE_DELTA_HEAD / USE_RESIDUAL_HEAD are set
    # by the orchestration shell script. Explicit True kwargs are never demoted.
    use_ray_pe = bool(use_ray_pe) or _env_bool("USE_RAY_PE", False)
    ray_pe_dim = _env_int("RAY_PE_DIM", ray_pe_dim)
    ray_pe_hidden = _env_int("RAY_PE_HIDDEN", ray_pe_hidden)
    # RayMap v2 representation. ``origin_dir`` (default) is the legacy 6D
    # [origin, dir] concat (bit-identical to the original ``ray_pe`` ablation).
    # ``plucker`` substitutes [dir, origin×dir] (still 6D); ``plucker_delta``
    # extends to 12D with per-frame delta vs the anchor frame. RAY_PE_DIM
    # must be 6 for origin_dir/plucker and 12 for plucker_delta.
    ray_pe_mode = os.environ.get("RAY_PE_MODE", ray_pe_mode) or "origin_dir"
    # DA3 RayMap v3 (2026-04-30): pose_conditioning_mode controls whether the
    # per-frame state token is dropped in favour of the dense per-patch raymap.
    # ``token+raymap`` (default) is byte-identical to every prior pilot.
    # ``raymap_only`` requires use_ray_pe=True (validated in the predictor).
    pose_conditioning_mode = os.environ.get(
        "POSE_CONDITIONING_MODE", pose_conditioning_mode
    ) or "token+raymap"
    action_token_mode = os.environ.get(
        "ACTION_TOKEN_MODE", action_token_mode
    ) or "transition"
    # RayMap v4c (2026-05-01): ray_visibility_features adds a rotation-only
    # warp correspondence block (uv_warp, valid, border) to plucker_pair.
    # Only valid with ray_pe_mode='plucker_pair'. See history doc §4.6.
    ray_visibility_features = os.environ.get(
        "RAY_VISIBILITY_FEATURES", ray_visibility_features
    ) or "none"
    if use_ray_pe and ray_pe_mode == "plucker_delta" and ray_pe_dim == 6:
        # Convenience: auto-promote to 12D when the user selects plucker_delta
        # without explicitly bumping RAY_PE_DIM. Avoids a confusing crash in
        # the predictor's __init__ when only RAY_PE_MODE was changed.
        ray_pe_dim = 12
    if use_ray_pe and ray_pe_mode == "plucker_pair":
        # Auto-promote ray_pe_dim for plucker_pair: 22 base + 4 if visibility
        # features enabled. Same convenience pattern as plucker_delta.
        expected_pair_dim = 22 + (4 if ray_visibility_features == "warp_plus_border" else 0)
        if ray_pe_dim in (6, 12):
            ray_pe_dim = expected_pair_dim
    use_delta_head = bool(use_delta_head) or _env_bool("USE_DELTA_HEAD", False)
    use_residual_head = bool(use_residual_head) or _env_bool("USE_RESIDUAL_HEAD", False)
    residual_head_depth = _env_int("RESIDUAL_HEAD_DEPTH", residual_head_depth)
    residual_head_ratio = _env_float("RESIDUAL_HEAD_RATIO", residual_head_ratio)
    use_completion_head = bool(use_completion_head) or _env_bool("USE_COMPLETION_HEAD", False)
    completion_head_depth = _env_int("COMPLETION_HEAD_DEPTH", completion_head_depth)
    completion_head_ratio = _env_float("COMPLETION_HEAD_RATIO", completion_head_ratio)
    use_four_layer_birth_head = bool(use_four_layer_birth_head) or _env_bool("USE_FOUR_LAYER_BIRTH_HEAD", False)
    completion_head_hidden = _env_int("COMPLETION_HEAD_HIDDEN", completion_head_hidden)
    # Rotation-only homography warp of context-frame tokens into the target
    # frame's grid. Diagnostic ablation; defaults to off so existing runs are
    # bit-identical. Requires use_intrinsics=True (per-frame K).
    warp_context_latents = bool(warp_context_latents) or _env_bool("WARP_CONTEXT_LATENTS", False)
    warp_context_padding_mode = os.environ.get("WARP_CONTEXT_PADDING_MODE", warp_context_padding_mode)
    # Path B-lite Step 1 — warp_mode + depth_probe_checkpoint env fallback.
    # ``WARP_MODE`` overrides the kwarg when set; ``DEPTH_PROBE_CHECKPOINT``
    # is a path to the checkpoint produced by tools/train_depth_probe_re10k.py
    # (required when warp_mode='projective_probe').
    warp_mode = os.environ.get("WARP_MODE", warp_mode) or "auto"
    depth_probe_checkpoint = (
        os.environ.get("DEPTH_PROBE_CHECKPOINT") or depth_probe_checkpoint
    )
    if depth_probe_checkpoint is not None:
        depth_probe_checkpoint = str(depth_probe_checkpoint)
    target_slot_mode = os.environ.get("TARGET_SLOT_MODE", target_slot_mode) or "mask"
    canvas_beta_init_valid = _env_float("CANVAS_BETA_INIT_VALID", canvas_beta_init_valid)
    canvas_beta_init_boundary = _env_float("CANVAS_BETA_INIT_BOUNDARY", canvas_beta_init_boundary)
    canvas_beta_init_oov = _env_float("CANVAS_BETA_INIT_OOV", canvas_beta_init_oov)
    canvas_use_mask_feat_embed = _env_bool("CANVAS_USE_MASK_FEAT_EMBED", canvas_use_mask_feat_embed)
    canvas_use_token_type_embed = _env_bool("CANVAS_USE_TOKEN_TYPE_EMBED", canvas_use_token_type_embed)
    canvas_use_target_step_embed = _env_bool("CANVAS_USE_TARGET_STEP_EMBED", canvas_use_target_step_embed)
    canvas_warp_mode = os.environ.get("CANVAS_WARP_MODE", canvas_warp_mode) or "raw"
    canvas_projector_gamma_init = _env_float("CANVAS_PROJECTOR_GAMMA_INIT", canvas_projector_gamma_init)
    canvas_norm_clip_ratio = _env_float("CANVAS_NORM_CLIP_RATIO", canvas_norm_clip_ratio)
    canvas_block_mask_enabled = bool(canvas_block_mask_enabled) or _env_bool("CANVAS_BLOCK_MASK_ENABLED", False)
    canvas_block_mask_min_rects = _env_int("CANVAS_BLOCK_MASK_MIN_RECTS", canvas_block_mask_min_rects)
    canvas_block_mask_max_rects = _env_int("CANVAS_BLOCK_MASK_MAX_RECTS", canvas_block_mask_max_rects)
    canvas_block_mask_min_frac = _env_float("CANVAS_BLOCK_MASK_MIN_FRAC", canvas_block_mask_min_frac)
    canvas_block_mask_max_frac = _env_float("CANVAS_BLOCK_MASK_MAX_FRAC", canvas_block_mask_max_frac)
    # Phase E1 correspondence-bias env-var fallback. Mirrors the warp pattern.
    correspondence_bias_enabled = bool(correspondence_bias_enabled) or _env_bool("CORR_BIAS_ENABLED", False)
    correspondence_bias_mode = os.environ.get("CORR_BIAS_MODE", correspondence_bias_mode)
    correspondence_bias_sigma_tokens = _env_float("CORR_BIAS_SIGMA_TOKENS", correspondence_bias_sigma_tokens)
    correspondence_bias_lambda_init = _env_float("CORR_BIAS_LAMBDA_INIT", correspondence_bias_lambda_init)
    if os.environ.get("CORR_BIAS_LEARNABLE") is not None:
        correspondence_bias_learnable = _env_bool("CORR_BIAS_LEARNABLE", correspondence_bias_learnable)
    apply_layers_env = _env_int_list("CORR_BIAS_APPLY_LAYERS")
    if apply_layers_env is not None:
        correspondence_bias_apply_layers = tuple(apply_layers_env)
    camera_ucpe_enabled = bool(camera_ucpe_enabled) or _env_bool("CAMERA_UCPE_ENABLED", False)
    camera_ucpe_gamma_init = _env_float("CAMERA_UCPE_GAMMA_INIT", camera_ucpe_gamma_init)
    camera_ucpe_layers_env = _env_int_list("CAMERA_UCPE_APPLY_LAYERS")
    if camera_ucpe_layers_env is not None:
        camera_ucpe_apply_layers = tuple(camera_ucpe_layers_env)
    predictor_adaln_enabled = bool(predictor_adaln_enabled) or _env_bool("PREDICTOR_ADALN_ENABLED", False)
    predictor_adaln_hidden = _env_int("PREDICTOR_ADALN_HIDDEN", predictor_adaln_hidden)
    coord_qk_enabled = bool(coord_qk_enabled) or _env_bool("COORD_QK_ENABLED", False)
    coord_qk_hidden = _env_int("COORD_QK_HIDDEN", coord_qk_hidden)
    coord_qk_freqs = _env_int("COORD_QK_FREQS", coord_qk_freqs)
    coord_qk_gamma_init = _env_float("COORD_QK_GAMMA_INIT", coord_qk_gamma_init)
    coord_qk_layers_env = _env_int_list("COORD_QK_APPLY_LAYERS")
    if coord_qk_layers_env is not None:
        coord_qk_apply_layers = tuple(coord_qk_layers_env)
    ray_qk_enabled = bool(ray_qk_enabled) or _env_bool("RAY_QK_ENABLED", False)
    ray_qk_gamma_init = _env_float("RAY_QK_GAMMA_INIT", ray_qk_gamma_init)
    ray_qk_layers_env = _env_int_list("RAY_QK_APPLY_LAYERS")
    if ray_qk_layers_env is not None:
        ray_qk_apply_layers = tuple(ray_qk_layers_env)
    target_motion_query_enabled = bool(target_motion_query_enabled) or _env_bool("TARGET_MOTION_QUERY_ENABLED", False)
    target_motion_query_hidden = _env_int("TARGET_MOTION_QUERY_HIDDEN", target_motion_query_hidden)
    target_motion_query_gamma_init = _env_float("TARGET_MOTION_QUERY_GAMMA_INIT", target_motion_query_gamma_init)
    logger.info(
        "init_video_model experiment flags: "
        f"encoder_backbone={encoder_backbone} (freeze={encoder_freeze}) "
        f"use_ray_pe={use_ray_pe} (mode={ray_pe_mode}, dim={ray_pe_dim}, hidden={ray_pe_hidden}, "
        f"visibility_features={ray_visibility_features}) "
        f"pose_conditioning_mode={pose_conditioning_mode} action_token_mode={action_token_mode} "
        f"use_delta_head={use_delta_head} "
        f"use_residual_head={use_residual_head} (depth={residual_head_depth}, ratio={residual_head_ratio}) "
        f"use_completion_head={use_completion_head} (depth={completion_head_depth}, ratio={completion_head_ratio}, "
        f"four_layer_birth={use_four_layer_birth_head}, hidden={completion_head_hidden}) "
        f"warp_context_latents={warp_context_latents} (padding={warp_context_padding_mode}) "
        f"warp_mode={warp_mode} depth_probe_checkpoint={depth_probe_checkpoint} "
        f"target_slot_mode={target_slot_mode} canvas_warp_mode={canvas_warp_mode} "
        f"canvas_block_mask_enabled={canvas_block_mask_enabled} "
        f"(rects={canvas_block_mask_min_rects}-{canvas_block_mask_max_rects}, "
        f"frac={canvas_block_mask_min_frac}-{canvas_block_mask_max_frac}) "
        f"correspondence_bias_enabled={correspondence_bias_enabled} "
        f"(mode={correspondence_bias_mode}, sigma={correspondence_bias_sigma_tokens}, "
        f"lambda_init={correspondence_bias_lambda_init}, learnable={correspondence_bias_learnable}, "
        f"apply_layers={tuple(correspondence_bias_apply_layers)}) "
        f"camera_ucpe_enabled={camera_ucpe_enabled} "
        f"(apply_layers={tuple(camera_ucpe_apply_layers)}, gamma_init={camera_ucpe_gamma_init}) "
        f"predictor_adaln_enabled={predictor_adaln_enabled} "
        f"(hidden={predictor_adaln_hidden}) "
        f"coord_qk_enabled={coord_qk_enabled} "
        f"(apply_layers={tuple(coord_qk_apply_layers)}, hidden={coord_qk_hidden}, "
        f"freqs={coord_qk_freqs}, gamma_init={coord_qk_gamma_init}) "
        f"ray_qk_enabled={ray_qk_enabled} "
        f"(apply_layers={tuple(ray_qk_apply_layers)}, gamma_init={ray_qk_gamma_init}) "
        f"target_motion_query_enabled={target_motion_query_enabled} "
        f"(hidden={target_motion_query_hidden}, gamma_init={target_motion_query_gamma_init})"
    )

    # ------------------------------------------------------------------
    # Encoder branch.
    #
    # ``vjepa21`` (default): build the canonical V-JEPA 2.1 ViT and pick
    # the canonical ``hierarchical_layers`` taps. Bit-identical to all
    # prior runs.
    #
    # Adapter backbones: build a frozen feature adapter at the same
    # crop_size and patch_size. The adapter exposes V-JEPA-compatible attributes
    # (``embed_dim``, ``num_heads``, ``hierarchical_layers``,
    # ``out_layers``, ``img_temporal_dim_size=1``) and a forward that
    # returns the same per-tap list shape, so the predictor side is
    # unchanged. ``encoder_freeze`` defaults to False here but is set by
    # the shell driver / Azure YAML to ``True`` for the MASt3R pilot
    # (matching the user-selected ``enc_lr_scale=0.0`` protocol).
    # ------------------------------------------------------------------
    if encoder_backbone in ("mast3r", "cradio"):
        if encoder_backbone == "mast3r":
            from app.vjepa_camera_re10k.mast3r_encoder import MASt3REncoderAdapter

            encoder = MASt3REncoderAdapter(
                img_size=crop_size,
                patch_size=patch_size,
                freeze=encoder_freeze,
            )
        else:
            from app.vjepa_camera_re10k.cradio_encoder import CRADIOEncoderAdapter

            _cradio_embed_dim = os.environ.get("CRADIO_EMBED_DIM", "").strip()
            encoder = CRADIOEncoderAdapter(
                model_version=os.environ.get("CRADIO_MODEL_VERSION", "c-radio_v4-h"),
                img_size=crop_size,
                patch_size=patch_size,
                embed_dim=int(_cradio_embed_dim) if _cradio_embed_dim else None,
                freeze=encoder_freeze,
                force_reload=_env_bool("CRADIO_FORCE_RELOAD", False),
                skip_validation=_env_bool("CRADIO_SKIP_VALIDATION", True),
            )
        # Validate the predictor's per-layer head split is consistent
        # with the adapter's chosen taps.
        if n_hierarchical_layers != len(encoder.hierarchical_layers):
            raise ValueError(
                f"n_hierarchical_layers={n_hierarchical_layers} but the {encoder_backbone} "
                f"adapter has hierarchical_layers={encoder.hierarchical_layers} "
                f"(len={len(encoder.hierarchical_layers)}). They must match "
                f"because the predictor splits target features by this count."
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
            correspondence_bias_enabled=correspondence_bias_enabled,
            correspondence_bias_mode=correspondence_bias_mode,
            correspondence_bias_sigma_tokens=correspondence_bias_sigma_tokens,
            correspondence_bias_lambda_init=correspondence_bias_lambda_init,
            correspondence_bias_learnable=correspondence_bias_learnable,
            correspondence_bias_apply_layers=tuple(correspondence_bias_apply_layers),
            camera_ucpe_enabled=camera_ucpe_enabled,
            camera_ucpe_apply_layers=tuple(camera_ucpe_apply_layers),
            camera_ucpe_gamma_init=camera_ucpe_gamma_init,
            predictor_adaln_enabled=predictor_adaln_enabled,
            predictor_adaln_hidden=predictor_adaln_hidden,
            coord_qk_enabled=coord_qk_enabled,
            coord_qk_apply_layers=tuple(coord_qk_apply_layers),
            coord_qk_hidden=coord_qk_hidden,
            coord_qk_freqs=coord_qk_freqs,
            coord_qk_gamma_init=coord_qk_gamma_init,
            ray_qk_enabled=ray_qk_enabled,
            ray_qk_apply_layers=tuple(ray_qk_apply_layers),
            ray_qk_gamma_init=ray_qk_gamma_init,
            target_motion_query_enabled=target_motion_query_enabled,
            target_motion_query_hidden=target_motion_query_hidden,
            target_motion_query_gamma_init=target_motion_query_gamma_init,
        )

        encoder.to(device)
        predictor.to(device)
        if os.environ.get("VJEPA_VERBOSE_MODEL_REPR", "0") == "1":
            logger.info(encoder)
            logger.info(predictor)

        n_enc_total = sum(p.numel() for p in encoder.parameters())
        n_enc_trainable = sum(p.numel() for p in encoder.parameters() if p.requires_grad)
        logger.info(
            f"Encoder ({encoder_backbone}): total={n_enc_total:,}  trainable={n_enc_trainable:,}"
        )
        logger.info(f"Predictor number of parameters: {sum(p.numel() for p in predictor.parameters() if p.requires_grad):,}")

        return encoder, predictor

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
        correspondence_bias_enabled=correspondence_bias_enabled,
        correspondence_bias_mode=correspondence_bias_mode,
        correspondence_bias_sigma_tokens=correspondence_bias_sigma_tokens,
        correspondence_bias_lambda_init=correspondence_bias_lambda_init,
        correspondence_bias_learnable=correspondence_bias_learnable,
        correspondence_bias_apply_layers=tuple(correspondence_bias_apply_layers),
        camera_ucpe_enabled=camera_ucpe_enabled,
        camera_ucpe_apply_layers=tuple(camera_ucpe_apply_layers),
        camera_ucpe_gamma_init=camera_ucpe_gamma_init,
        predictor_adaln_enabled=predictor_adaln_enabled,
        predictor_adaln_hidden=predictor_adaln_hidden,
        coord_qk_enabled=coord_qk_enabled,
        coord_qk_apply_layers=tuple(coord_qk_apply_layers),
        coord_qk_hidden=coord_qk_hidden,
        coord_qk_freqs=coord_qk_freqs,
        coord_qk_gamma_init=coord_qk_gamma_init,
        ray_qk_enabled=ray_qk_enabled,
        ray_qk_apply_layers=tuple(ray_qk_apply_layers),
        ray_qk_gamma_init=ray_qk_gamma_init,
        target_motion_query_enabled=target_motion_query_enabled,
        target_motion_query_hidden=target_motion_query_hidden,
        target_motion_query_gamma_init=target_motion_query_gamma_init,
    )

    encoder.to(device)
    predictor.to(device)
    if os.environ.get("VJEPA_VERBOSE_MODEL_REPR", "0") == "1":
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
    corrector=None,
    wd=1e-6,
    final_wd=1e-6,
    final_lr=0.0,
    mixed_precision=False,
    betas=(0.9, 0.999),
    eps=1e-8,
    zero_init_bias_wd=True,
    enc_lr_scale=1.0,
):
    # Filter on requires_grad so frozen modules (e.g. encoder + predictor in
    # the corrector's frozen-predictor mode) do not contribute empty groups
    # — AdamW raises on empty param groups in some configurations.
    def _decay_params(module):
        return [
            p for n, p in module.named_parameters()
            if p.requires_grad and ("bias" not in n) and (len(p.shape) != 1)
        ]

    def _no_decay_params(module):
        return [
            p for n, p in module.named_parameters()
            if p.requires_grad and (("bias" in n) or (len(p.shape) == 1))
        ]

    param_groups = []
    enc_decay = _decay_params(encoder)
    if enc_decay:
        param_groups.append({"params": enc_decay, "lr_scale": enc_lr_scale})
    pred_decay = _decay_params(predictor)
    if pred_decay:
        param_groups.append({"params": pred_decay})
    enc_nodecay = _no_decay_params(encoder)
    if enc_nodecay:
        param_groups.append({
            "params": enc_nodecay,
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
            "lr_scale": enc_lr_scale,
        })
    pred_nodecay = _no_decay_params(predictor)
    if pred_nodecay:
        param_groups.append({
            "params": pred_nodecay,
            "WD_exclude": zero_init_bias_wd,
            "weight_decay": 0,
        })
    if corrector is not None:
        # Phase C — latent corrector: train at the same base lr as the
        # predictor but with no decay on residual scales (treated as 1-D bias).
        corr_decay = _decay_params(corrector)
        if corr_decay:
            param_groups.append({"params": corr_decay})
        corr_nodecay = _no_decay_params(corrector)
        if corr_nodecay:
            param_groups.append({
                "params": corr_nodecay,
                "WD_exclude": zero_init_bias_wd,
                "weight_decay": 0,
            })

    if not param_groups:
        raise RuntimeError(
            "init_opt: no trainable parameters found across encoder, predictor, "
            "and corrector. Check requires_grad flags and corrector mode."
        )

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
