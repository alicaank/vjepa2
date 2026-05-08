"""Inference-time latent corrector for camera-conditioned V-JEPA rollout.

Phase C of the Transport-Correct-Reverse (TCR) design; see
``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` §5.

Purpose
-------
A small per-layer residual module that re-projects a self-fed predicted V-JEPA
2.1 latent back toward the target-encoder distribution before it becomes the
next rollout step's input. The corrector is applied **only** to the
self-feeding slot in the closed-loop rollout; the loss-supervised forward
prediction is never modified in place.

Why per-layer
-------------
The hierarchical V-JEPA target concatenates ``n_hierarchical_layers`` encoder
layers along the channel axis. A monolithic MLP over ``n_layers * embed_dim``
mixes layer semantics and tends to over-smooth (deeper V-JEPA layers carry
slow-varying semantic content; shallow layers carry high-frequency spatial
detail). Per-layer blocks keep them separable.

Conditioning
------------
The corrector is mildly pose-aware (``pose_mag``: translation magnitude only)
and rollout-step-aware (``step_idx``: 0 .. K-1). Translation magnitude is
chosen rather than the full SE(3) pose to avoid duplicating the predictor's
job; the corrector should learn "under big motion, correct more," not relearn
the dynamics.

Initialization
--------------
Per-layer residual scales are initialized near zero so the first epoch is a
no-op. This makes the ``corrector.enabled = false`` ablation a clean control:
turning the corrector off (or never having trained it) has no effect on the
predictor.
"""

from __future__ import annotations

from typing import Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


class LatentCorrector(nn.Module):
    """Per-layer, token-wise corrective residual on a self-fed predicted latent.

    Args:
        n_layers: Number of V-JEPA hierarchical layers concatenated along the
            channel axis of the predicted latent (typically 4 for V-JEPA 2.1
            ViT-L 384).
        embed_dim: Per-layer embedding dimension (typically 1024 for ViT-L).
            The total channel width is ``n_layers * embed_dim``.
        hidden: Width of the per-layer block's hidden representation.
        step_embed_dim: Width of the rollout-step embedding.
        pose_embed_dim: Width of the pose-magnitude embedding.
        residual_scale_init: Initial value of the per-layer residual scale.
            Should be small (default 0.01) so the first epoch is approximately
            a no-op and the ``corrector.enabled = false`` ablation is clean.
        residual_scale_max: Soft cap on ``|scale[l]|`` used at forward time
            (clamped via ``torch.clamp``). Prevents the corrector from
            dominating the predictor output during early training.
        max_rollout_steps: Maximum supported rollout horizon. Determines the
            size of the step embedding table.

    Forward signature:
        forward(z_raw, step_idx, pose_mag) -> z_corr
        - z_raw:    (B, HW, n_layers * embed_dim) predicted latent.
        - step_idx: (B,) long tensor in [0, max_rollout_steps).
        - pose_mag: (B,) float tensor; translation magnitude of the action
            applied to produce ``z_raw`` (i.e. ``||a[..., :3]||``).
        - z_corr:   same shape as z_raw.

    Output:
        ``z_corr = z_raw + sum_l clamp(scale[l]) * block_l(z_raw_l, cond)``
    """

    def __init__(
        self,
        n_layers: int,
        embed_dim: int,
        hidden: int = 512,
        step_embed_dim: int = 32,
        pose_embed_dim: int = 32,
        residual_scale_init: float = 0.01,
        residual_scale_max: float = 0.25,
        max_rollout_steps: int = 32,
    ) -> None:
        super().__init__()
        if n_layers <= 0:
            raise ValueError(f"n_layers must be positive; got {n_layers}")
        if embed_dim <= 0:
            raise ValueError(f"embed_dim must be positive; got {embed_dim}")

        self.n_layers = int(n_layers)
        self.embed_dim = int(embed_dim)
        self.residual_scale_max = float(residual_scale_max)
        self.max_rollout_steps = int(max_rollout_steps)

        # Conditioning: rollout-step + pose-magnitude embeddings.
        self.step_embed = nn.Embedding(self.max_rollout_steps, step_embed_dim)
        self.pose_mlp = nn.Sequential(
            nn.Linear(1, pose_embed_dim),
            nn.GELU(),
            nn.Linear(pose_embed_dim, pose_embed_dim),
        )
        cond_dim = step_embed_dim + pose_embed_dim

        # One block per V-JEPA hierarchical layer.
        self.layer_blocks = nn.ModuleList(
            [
                nn.Sequential(
                    nn.Linear(self.embed_dim + cond_dim, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, hidden),
                    nn.GELU(),
                    nn.Linear(hidden, self.embed_dim),
                )
                for _ in range(self.n_layers)
            ]
        )

        # Per-layer learned residual scale. Init near zero so first epoch
        # is approximately a no-op (clean ablation control).
        self.scale = nn.Parameter(
            torch.full((self.n_layers,), float(residual_scale_init))
        )

        # Initialize the last linear layer of each block to small magnitude
        # so the very first forward pass produces a near-zero residual even
        # before the residual scale has trained.
        for block in self.layer_blocks:
            last_linear = block[-1]
            assert isinstance(last_linear, nn.Linear)
            nn.init.normal_(last_linear.weight, mean=0.0, std=1e-3)
            if last_linear.bias is not None:
                nn.init.zeros_(last_linear.bias)

    @torch.jit.ignore
    def _effective_scale(self) -> torch.Tensor:
        """Return per-layer residual scales clamped to ``[-max, max]``."""
        return self.scale.clamp(-self.residual_scale_max, self.residual_scale_max)

    def forward(
        self,
        z_raw: torch.Tensor,
        step_idx: torch.Tensor,
        pose_mag: torch.Tensor,
    ) -> torch.Tensor:
        if z_raw.dim() != 3:
            raise ValueError(
                f"z_raw must be (B, HW, n_layers*embed_dim); got shape "
                f"{tuple(z_raw.shape)}"
            )
        B, HW, D = z_raw.shape
        expected = self.n_layers * self.embed_dim
        if D != expected:
            raise ValueError(
                f"z_raw has channel dim {D} but corrector expects "
                f"n_layers*embed_dim = {self.n_layers}*{self.embed_dim} = "
                f"{expected}"
            )
        if step_idx.dim() != 1 or step_idx.shape[0] != B:
            raise ValueError(
                f"step_idx must be (B,); got {tuple(step_idx.shape)} (B={B})"
            )
        if pose_mag.dim() != 1 or pose_mag.shape[0] != B:
            raise ValueError(
                f"pose_mag must be (B,); got {tuple(pose_mag.shape)} (B={B})"
            )

        # Clamp step_idx to embedding table size; conservative no-fail behavior
        # in case a rollout exceeds max_rollout_steps at eval time.
        step_idx_clamped = step_idx.clamp(0, self.max_rollout_steps - 1)

        s = self.step_embed(step_idx_clamped)                  # (B, step_embed_dim)
        p = self.pose_mlp(pose_mag.to(z_raw.dtype).unsqueeze(-1))  # (B, pose_embed_dim)
        cond = torch.cat([s.to(z_raw.dtype), p], dim=-1)       # (B, cond_dim)
        cond = cond.unsqueeze(1).expand(B, HW, -1)             # (B, HW, cond_dim)

        chunks = z_raw.chunk(self.n_layers, dim=-1)            # n_layers x (B, HW, embed_dim)
        if len(chunks) != self.n_layers:
            raise RuntimeError(
                f"chunked into {len(chunks)} pieces but expected "
                f"{self.n_layers}; check embed_dim divisibility."
            )

        scales = self._effective_scale()                       # (n_layers,)
        outs = []
        for l, c in enumerate(chunks):
            x = torch.cat([c, cond], dim=-1)                   # (B, HW, embed_dim + cond_dim)
            delta = self.layer_blocks[l](x)                    # (B, HW, embed_dim)
            outs.append(c + scales[l] * delta)
        return torch.cat(outs, dim=-1)

    @torch.no_grad()
    def per_layer_norm_stats(
        self,
        z_raw: torch.Tensor,
        step_idx: torch.Tensor,
        pose_mag: torch.Tensor,
    ) -> dict:
        """Return diagnostic norms per V-JEPA layer for a single forward pass.

        Useful for the monitoring signals listed in
        ``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` §5.1: input norm, output
        norm, and per-layer residual delta norm. All values are scalar floats
        averaged over the batch and tokens.

        Returns a dict with keys
            - ``input_norm_l{l}``  for l in 0..n_layers-1
            - ``output_norm_l{l}``
            - ``delta_norm_l{l}``
            - ``scale_l{l}`` — current effective per-layer scale
        """
        # Run forward in eval-equivalent mode but keep dropout off (we have no
        # dropout). Cheap to repeat the math because this is monitoring only.
        z_corr = self.forward(z_raw, step_idx, pose_mag)
        in_chunks = z_raw.chunk(self.n_layers, dim=-1)
        out_chunks = z_corr.chunk(self.n_layers, dim=-1)
        scales = self._effective_scale()

        stats = {}
        for l in range(self.n_layers):
            in_l = in_chunks[l]
            out_l = out_chunks[l]
            delta_l = out_l - in_l
            stats[f"input_norm_l{l}"] = float(in_l.norm(dim=-1).mean().item())
            stats[f"output_norm_l{l}"] = float(out_l.norm(dim=-1).mean().item())
            stats[f"delta_norm_l{l}"] = float(delta_l.norm(dim=-1).mean().item())
            stats[f"scale_l{l}"] = float(scales[l].item())
        return stats


def build_latent_corrector(
    n_hierarchical_layers: int,
    embed_dim: int,
    cfg: Optional[dict] = None,
) -> LatentCorrector:
    """Construct a :class:`LatentCorrector` from a (possibly empty) config dict.

    Defaults match the values in
    ``docs/TRANSPORT_CORRECT_REVERSE_DESIGN.md`` §5.5.
    """
    cfg = cfg or {}
    return LatentCorrector(
        n_layers=int(n_hierarchical_layers),
        embed_dim=int(embed_dim),
        hidden=int(cfg.get("hidden", 512)),
        step_embed_dim=int(cfg.get("step_embed_dim", 32)),
        pose_embed_dim=int(cfg.get("pose_embed_dim", 32)),
        residual_scale_init=float(cfg.get("residual_scale_init", 0.01)),
        residual_scale_max=float(cfg.get("residual_scale_max", 0.25)),
        max_rollout_steps=int(cfg.get("max_rollout_steps", 32)),
    )
