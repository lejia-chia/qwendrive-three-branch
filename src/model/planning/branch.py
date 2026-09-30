"""Differentiable clean-endpoint planning supervision from shared S+T K/V.

Official PlanningExpert is reused without modifying the upstream repository.
The released code provides inference, not its training recipe: masked endpoint MSE
with a linear flow path is an explicit supervised baseline choice here.
"""
from __future__ import annotations

import torch
from qwen_drive.trajectory import normalize_history, normalize_trajectory, denormalize_trajectory

def masked_endpoint_mse(prediction, target, valid, scale):
    """Per-valid-coordinate normalized MSE; heading residual is periodic."""
    if prediction.shape != target.shape or prediction.ndim != 3 or prediction.shape[-1] != 3:
        raise ValueError("endpoint/GT shape must agree at [B,50,3]")
    if valid.shape != prediction.shape[:2] or not torch.isin(valid, valid.new_tensor([0, 1])).all():
        raise ValueError("trajectory mask must be binary [B,50]")
    keep = valid.bool()
    if not keep.any() or not torch.isfinite(target[keep]).all():
        raise ValueError("no valid GT or nonfinite valid GT")
    if not torch.isfinite(prediction[keep]).all():
        raise ValueError("nonfinite predicted endpoint")
    delta = prediction.float()[keep] - target.detach().float()[keep]
    yaw = delta[:, 2] * scale[2]
    yaw = torch.atan2(yaw.sin(), yaw.cos()) / scale[2]
    return torch.cat((delta[:, :2], yaw[:, None]), dim=-1).square().mean()


def planning_supervision(expert, *, scene_kv, anchor, inputs, trajectory, valid,
                         scale, noise, flow_time):
    """Training-only GT -> noisy waypoint input; GT never reaches S/T/Future.

    x_t = (1-t) noise + t normalize(GT), endpoint target = normalize(GT).
    Invalid GT points are sanitized before constructing the noisy sequence.
    This is an endpoint-parameterized flow baseline, not an official RL loss.
    """
    gt = trajectory.detach().float()
    mask = valid.detach()
    if gt.shape != noise.shape or mask.shape != gt.shape[:2]:
        raise ValueError("noise/GT/mask geometry differs")
    if flow_time.shape != (gt.shape[0],) or not torch.isfinite(flow_time).all() or not ((flow_time >= 0) & (flow_time < 1)).all():
        raise ValueError("flow time must be finite in [0,1)")
    if not torch.isfinite(noise).all() or not torch.isin(mask, mask.new_tensor([0,1])).all():
        raise ValueError("invalid noise/mask")
    keep = mask.bool()
    if not keep.any() or not torch.isfinite(gt[keep]).all():
        raise ValueError("invalid GT supervision")
    gt = torch.where(keep[..., None], gt, torch.zeros_like(gt))
    clean = normalize_trajectory(gt, scale)
    t = flow_time[:, None, None]
    noisy = (1-t) * noise.detach().float() + t * clean
    history = normalize_history(inputs['history'].float(), scale)
    queries = expert.encode_history(history, inputs['nav_command'],
                                    inputs['history_velocity'], inputs['history_acceleration'])
    endpoint = expert.predict_endpoint(noisy, flow_time, queries, scene_kv, anchor,
                                       inputs['nav_command'], inputs['ego_status'])
    loss = masked_endpoint_mse(endpoint, clean, mask, scale)
    return {'loss':loss, 'endpoint_normalized':endpoint,
            'trajectory':denormalize_trajectory(endpoint, scale),
            'noisy_waypoints':noisy.detach(), 'flow_time':flow_time.detach()}


@torch.no_grad()
def sample_planning(expert, *, scene_kv, anchor, inputs, scale, noise,
                    num_steps=10, min_one_minus_t=.1):
    """Inference endpoint integration: no GT trajectory argument exists."""
    normalized = expert.sample(scene_cache=scene_kv, position_anchor=anchor,
        history=normalize_history(inputs['history'].float(), scale),
        history_velocity=inputs['history_velocity'], history_acceleration=inputs['history_acceleration'],
        nav_command=inputs['nav_command'], ego_status=inputs['ego_status'], noise=noise,
        num_steps=num_steps, min_one_minus_t=min_one_minus_t)
    return denormalize_trajectory(normalized, scale)
