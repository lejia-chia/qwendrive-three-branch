#!/usr/bin/env python
"""Reason scoring branch: S+T reason hidden + FIXED physical candidate features -> p.

Contract
--------
    input       reason hidden over the reason span of S + T      [T, 2560]
                fixed physical features of each candidate        [K, F]
    output      one score per candidate, p = softmax(scores)

The GT is one candidate; its identity and GT-relative distances are not head
inputs. The features are functions of each candidate trajectory alone, and the head
is shared across candidates, so the scores are permutation-equivariant: shuffling
the candidate list moves each score with its own candidate.  No trainable
trajectory encoder is used (the retired contrastive encoder collapsed; the fixed
features are deliberately not learned).

The target q is supervision, not input: it exists only offline (data/q_targets/v2)
and is produced by the frozen rules in ``rules.json`` of that artifact.

Features (F = 72, all in the ego frame, metres and seconds)
-----------------------------------------------------------
    x, y at 10 horizons (0.5 .. 5.0 s)                          20
    vx, vy at the same horizons (finite difference over 0.1 s)  20
    sin/cos of the candidate's own heading at the horizons      20
    cumulative arc length at the horizons                       10
    valid-frame fraction inside 4 s / inside 5 s                 2
Frames that are invalid at a horizon carry the last valid value forward (documented
in the feature docstring); the fractions expose how much was carried.
"""
from __future__ import annotations

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

DT = 0.1
STEPS = 50
HORIZONS = tuple(range(5, 51, 5))            # 0.5 s .. 5.0 s, ten horizons
FEATURE_DIM = 7 * len(HORIZONS) + 2      # 72: xy(2) + v(2) + sin/cos(2) + arc(1) per horizon, + 2 fractions
HORIZON_4S = 40

FAMILY_NAMES = ("gt", "longitudinal", "lateral", "timing", "cross_scene")


# ----------------------------------------------------------------------- features

def candidate_features(trajectory: np.ndarray, mask: np.ndarray,
                       *, horizons: tuple[int, ...] = HORIZONS,
                       horizon_4s: int = HORIZON_4S) -> np.ndarray:
    """The fixed feature vector of one candidate.  Deterministic, GT-free.

    ``trajectory`` is [50, 3] (x, y, heading) in the ego frame, ``mask`` [50] with
    1 = valid.  At each horizon the state is taken from the last valid frame at or
    before it (carry-forward), so a truncated candidate still yields finite
    features; the two validity fractions say how much of the horizon was carried.
    """
    trajectory = np.asarray(trajectory, dtype=np.float64)
    raw_mask = np.asarray(mask, dtype=np.float64)
    if not np.isfinite(raw_mask).all() or not np.isin(raw_mask, (0, 1)).all():
        raise ValueError("candidate mask must be finite and binary")
    mask = raw_mask > 0.5
    if trajectory.shape != (STEPS, 3) or mask.shape != (STEPS,):
        raise ValueError(f"candidate shape {trajectory.shape} / {mask.shape} is not 50x3 / 50")
    if not mask.any():
        raise ValueError("candidate has no valid frame")
    if not np.isfinite(trajectory[mask]).all():
        raise ValueError("nonfinite valid candidate coordinate")
    valid_index = np.where(mask)[0]
    last_valid_at_or_before = np.maximum.accumulate(np.where(mask, np.arange(STEPS), -1))
    last_valid_at_or_before = np.maximum(last_valid_at_or_before, valid_index[0])
    positions, velocities, headings, arclengths = [], [], [], []
    # Compute arc length on the same carried states as position features.
    # Invalid coordinates, including NaN, must never affect any feature.
    carried = trajectory[last_valid_at_or_before]
    step_distance = np.linalg.norm(np.diff(carried[:, :2], axis=0), axis=1)
    cumulative = np.concatenate([[0.0], np.cumsum(step_distance)])
    for horizon in horizons:
        index = last_valid_at_or_before[horizon - 1]
        positions.append(trajectory[index, :2])
        if index >= 1 and mask[index - 1] and mask[index]:
            velocities.append((trajectory[index, :2] - trajectory[index - 1, :2]) / DT)
        else:
            velocities.append(np.zeros(2))
        headings.append(trajectory[index, 2])
        arclengths.append(cumulative[index])
    positions = np.asarray(positions).reshape(-1)                     # 20
    velocities = np.asarray(velocities).reshape(-1)                   # 20
    angles = np.asarray(headings)
    heading_features = np.stack([np.sin(angles), np.cos(angles)], axis=1).reshape(-1)
    fractions = np.array([mask[:horizon_4s].mean(), mask.mean()])
    return np.concatenate([positions, velocities, heading_features,
                           np.asarray(arclengths), fractions]).astype(np.float32)


def features_for_record(record: dict, *, row: int) -> np.ndarray:
    """[K, F] features for the candidates of one record (padded slots included)."""
    count = int(record["counts"][row])
    out = np.zeros((record["trajectories"].shape[1], FEATURE_DIM), dtype=np.float32)
    for slot in range(count):
        out[slot] = candidate_features(record["trajectories"][row, slot],
                                       record["masks"][row, slot])
    return out


def all_features(arrays: dict) -> np.ndarray:
    """[N, K, F] features for a whole split (padding slots stay zero)."""
    n, k = arrays["trajectories"].shape[0], arrays["trajectories"].shape[1]
    out = np.zeros((n, k, FEATURE_DIM), dtype=np.float32)
    for row in range(n):
        for slot in range(int(arrays["counts"][row])):
            out[row, slot] = candidate_features(arrays["trajectories"][row, slot],
                                                arrays["masks"][row, slot])
    return out


# --------------------------------------------------------------------------- head

class ReasonScoreHead(nn.Module):
    """A shared per-candidate scorer; the only new module (float32).

    ``reason_proj`` maps the pooled reason hidden to a width-``width`` vector,
    ``feature_proj`` maps the fixed features to width ``feature_width``; the two are
    concatenated per candidate and scored by a small MLP.  Because the MLP is shared
    across candidates and the pooling is order-free, the head is permutation
    equivariant by construction.
    """

    def __init__(self, hidden: int = 2560, feature_dim: int = FEATURE_DIM,
                 width: int = 256, feature_width: int = 128):
        super().__init__()
        self.reason_norm = nn.LayerNorm(hidden)
        self.reason_proj = nn.Linear(hidden, width)
        self.feature_norm = nn.LayerNorm(feature_dim)
        self.feature_proj = nn.Linear(feature_dim, feature_width)
        self.scorer = nn.Sequential(nn.Linear(width + feature_width, width), nn.GELU(),
                                    nn.Linear(width, 1))
        self.width, self.feature_width = width, feature_width

    def forward(self, reason_hidden: torch.Tensor, features: torch.Tensor,
                mask: torch.Tensor | None = None) -> torch.Tensor:
        dtype = self.reason_proj.weight.dtype
        hidden = self.reason_norm(reason_hidden.to(dtype))
        if mask is None:
            pooled = hidden.mean(dim=1)
        else:
            weights = mask.to(dtype).unsqueeze(-1)
            pooled = (hidden * weights).sum(1) / weights.sum(1).clamp(min=1.0)
        reason = self.reason_proj(pooled)                                   # [B, width]
        feature = self.feature_proj(self.feature_norm(features.to(dtype)))  # [B, K, fw]
        joined = torch.cat([reason.unsqueeze(1).expand(-1, feature.shape[1], -1),
                            feature], dim=-1)
        return self.scorer(joined).squeeze(-1)                              # [B, K]

    def probabilities(self, reason_hidden: torch.Tensor, features: torch.Tensor,
                      mask: torch.Tensor | None = None, valid: torch.Tensor | None = None):
        """(p, raw scores).  Invalid slots are masked to zero probability in ``p``.

        The raw scores are returned unmasked so callers can compare them numerically
        (a masked score is -inf, which poisons difference checks).
        """
        scores = self(reason_hidden, features, mask).float()
        if valid is not None and not valid.bool().any(dim=-1).all():
            raise ValueError("probabilities require a live candidate in every row")
        masked = scores.masked_fill(~valid.bool(), float("-inf")) if valid is not None \
            else scores
        return torch.softmax(masked, dim=-1), scores


# ------------------------------------------------------------------------- losses

def kl_divergence(scores: torch.Tensor, q: torch.Tensor, valid: torch.Tensor) -> torch.Tensor:
    """Per-record KL(q || softmax(scores)) over the valid slots (0 where q is 0).

    Used where a proper distribution is guaranteed; the training and evaluation
    paths go through :func:`kl_from_probabilities` so both see the SAME masking.
    """
    valid = valid.bool()
    if not valid.any(dim=-1).all():
        raise ValueError("KL requires at least one valid candidate per row")
    q = q.detach().float()
    if (not torch.isfinite(q).all() or (q < 0).any()
            or (q.masked_select(~valid) != 0).any()
            or not torch.allclose(q.sum(-1), torch.ones_like(q.sum(-1)), atol=1e-6)):
        raise ValueError("invalid target distribution")
    if not torch.isfinite(scores.masked_select(valid)).all():
        raise ValueError("nonfinite live score")
    log_p = F.log_softmax(scores.float().masked_fill(~valid, float("-inf")), dim=-1)
    positive = valid.bool() & (q > 0)
    terms = torch.zeros_like(q)
    terms[positive] = q[positive] * (q[positive].log() - log_p[positive])
    return terms.sum(dim=-1)


