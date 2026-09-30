"""Native-grid Future prediction from the single differentiable S+T VLM pass.

The spatial grid is the *unpooled* output grid of the frozen QwenDrive vision
tower.  No 2x2/8x8 resampling or fixed spatial-token count occurs here.  Query
order and output order are camera, future time, row, column, channel.

Every layer cross-attends to one full-attention K/V source covering S+T.  Query
chunks bound attention workspace; windowed spatial attention and attention
along ten future times mix the grid without quadratic global self-attention. The
scene K/V must remain attached to the VLM graph during joint training.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F


@dataclass(frozen=True)
class NativeFutureConfig:
    cameras: int = 3
    future_steps: int = 10
    hidden_size: int = 512
    layers: int = 2
    source_indices: tuple[int, ...] = (3, 7)
    attention_heads: int = 8
    kv_heads: int = 4
    head_dim: int = 256
    ffn_size: int = 1536
    out_channels: int = 2560
    query_chunk: int = 256
    spatial_window: int = 4
    rotary_fraction: float = 0.25
    rope_theta: float = 1e7
    use_scene_kv: bool = True

    def __post_init__(self) -> None:
        if self.cameras != 3 or self.future_steps != 10:
            raise ValueError("native target contract requires 3 cameras and 10 future times")
        if self.layers < 1 or len(self.source_indices) != self.layers:
            raise ValueError("one K/V source index is required for each Future layer")
        if len(set(self.source_indices)) != len(self.source_indices) or any(
            not 0 <= index < 8 for index in self.source_indices
        ):
            raise ValueError("source indices must be distinct members of the eight exported layers")
        if self.attention_heads % self.kv_heads or self.head_dim != 256:
            raise ValueError("GQA grouping or QwenDrive K/V head dimension is invalid")
        if self.query_chunk < 1:
            raise ValueError("query_chunk must be positive")
        if self.spatial_window < 1 or self.hidden_size % self.attention_heads:
            raise ValueError("invalid spatial window or self-attention head grouping")
        rotary_dim = int(self.head_dim * self.rotary_fraction)
        if rotary_dim < 2 or rotary_dim % 2:
            raise ValueError("rotary head dimension must be positive and even")


def _rotate_half(x: torch.Tensor) -> torch.Tensor:
    left, right = x.chunk(2, dim=-1)
    return torch.cat((-right, left), dim=-1)


def _rotary(query: torch.Tensor, positions: torch.Tensor, *, fraction: float,
            theta: float) -> torch.Tensor:
    """Apply 1-D rotary continuation to the query prefix; scene K is post-RoPE."""
    dim = int(query.shape[-1] * fraction)
    inv = theta ** (-torch.arange(0, dim, 2, device=query.device,
                                 dtype=torch.float32) / dim)
    angle = positions.float().unsqueeze(-1) * inv
    frequencies = torch.cat((angle, angle), dim=-1).unsqueeze(2)
    first, rest = query[..., :dim], query[..., dim:]
    rotated = first * frequencies.cos().to(first.dtype) + _rotate_half(first) * frequencies.sin().to(first.dtype)
    return torch.cat((rotated, rest), dim=-1)


class NativeFutureLayer(nn.Module):
    def __init__(self, config: NativeFutureConfig) -> None:
        super().__init__()
        self.config = config
        d = config.hidden_size
        self.query_norm = nn.LayerNorm(d)
        self.q_proj = nn.Linear(d, config.attention_heads * config.head_dim, bias=False)
        self.gate_proj = nn.Linear(d, config.attention_heads * config.head_dim)
        self.out_proj = nn.Linear(config.attention_heads * config.head_dim, d, bias=False)
        self.spatial_norm = nn.LayerNorm(d)
        self.spatial_attention = nn.MultiheadAttention(
            d, config.attention_heads, batch_first=True
        )
        self.temporal_norm = nn.LayerNorm(d)
        self.temporal_attention = nn.MultiheadAttention(
            d, config.attention_heads, batch_first=True
        )
        self.ffn_norm = nn.LayerNorm(d)
        self.ffn_up = nn.Linear(d, config.ffn_size * 2, bias=False)
        self.ffn_down = nn.Linear(config.ffn_size, d, bias=False)

    def forward(self, hidden: torch.Tensor, kv: tuple[torch.Tensor, torch.Tensor],
                positions: torch.Tensor, grid: tuple[int, int]) -> torch.Tensor:
        cfg = self.config
        batch, count, width = hidden.shape
        if cfg.use_scene_kv:
            key, value = kv
            key = key.transpose(1, 2)
            value = value.transpose(1, 2)
            chunks = []
            for start in range(0, count, cfg.query_chunk):
                end = min(start + cfg.query_chunk, count)
                source = self.query_norm(hidden[:, start:end])
                q = self.q_proj(source).reshape(batch, end - start,
                                                cfg.attention_heads, cfg.head_dim)
                q = _rotary(q, positions[:, start:end], fraction=cfg.rotary_fraction,
                            theta=cfg.rope_theta).transpose(1, 2)
                attended = F.scaled_dot_product_attention(
                    q, key.to(q.dtype), value.to(q.dtype), enable_gqa=True
                ).transpose(1, 2).reshape(batch, end - start, -1)
                gate = torch.sigmoid(self.gate_proj(source))
                chunks.append(self.out_proj(attended * gate))
            hidden = hidden + torch.cat(chunks, dim=1)

        height, width_grid = grid
        window = cfg.spatial_window
        pad_h = (-height) % window
        pad_w = (-width_grid) % window
        padded_h, padded_w = height + pad_h, width_grid + pad_w
        rows, cols = padded_h // window, padded_w // window
        spatial = self.spatial_norm(hidden).reshape(
            batch, cfg.cameras, cfg.future_steps, height, width_grid, width
        )
        spatial = F.pad(spatial, (0, 0, 0, pad_w, 0, pad_h))
        spatial = spatial.reshape(batch, cfg.cameras, cfg.future_steps,
                                  rows, window, cols, window, width)
        spatial = spatial.permute(0, 1, 2, 3, 5, 4, 6, 7).reshape(-1, window * window, width)
        invalid = torch.ones(padded_h, padded_w, dtype=torch.bool, device=hidden.device)
        invalid[:height, :width_grid] = False
        invalid = invalid.reshape(rows, window, cols, window).permute(0, 2, 1, 3)
        invalid = invalid.reshape(1, 1, 1, rows, cols, window * window).expand(
            batch, cfg.cameras, cfg.future_steps, -1, -1, -1
        ).reshape(-1, window * window)
        spatial, _ = self.spatial_attention(
            spatial, spatial, spatial, key_padding_mask=invalid, need_weights=False
        )
        spatial = spatial.reshape(batch, cfg.cameras, cfg.future_steps,
                                  rows, cols, window, window, width)
        spatial = spatial.permute(0, 1, 2, 3, 5, 4, 6, 7).reshape(
            batch, cfg.cameras, cfg.future_steps, padded_h, padded_w, width
        )[:, :, :, :height, :width_grid].reshape(batch, count, width)
        hidden = hidden + spatial

        temporal = self.temporal_norm(hidden).reshape(
            batch, cfg.cameras, cfg.future_steps, height, width_grid, width
        ).permute(0, 1, 3, 4, 2, 5).reshape(-1, cfg.future_steps, width)
        temporal, _ = self.temporal_attention(
            temporal, temporal, temporal, need_weights=False
        )
        temporal = temporal.reshape(batch, cfg.cameras, height, width_grid,
                                    cfg.future_steps, width).permute(0, 1, 4, 2, 3, 5)
        hidden = hidden + temporal.reshape(batch, count, width)
        gate, up = self.ffn_up(self.ffn_norm(hidden)).chunk(2, dim=-1)
        return hidden + self.ffn_down(F.silu(gate) * up)


class NativeFutureBranch(nn.Module):
    """Predict [B,3,10,H,W,2560] for the encoder's runtime native H,W grid."""

    def __init__(self, config: NativeFutureConfig | None = None) -> None:
        super().__init__()
        self.config = config or NativeFutureConfig()
        cfg = self.config
        self.base = nn.Parameter(torch.zeros(cfg.hidden_size))
        self.camera = nn.Embedding(cfg.cameras, cfg.hidden_size)
        self.time = nn.Embedding(cfg.future_steps, cfg.hidden_size)
        self.xy = nn.Sequential(nn.Linear(2, cfg.hidden_size), nn.SiLU(),
                                nn.Linear(cfg.hidden_size, cfg.hidden_size))
        self.blocks = nn.ModuleList(NativeFutureLayer(cfg) for _ in range(cfg.layers))
        self.final_norm = nn.LayerNorm(cfg.hidden_size)
        self.out_proj = nn.Linear(cfg.hidden_size, cfg.out_channels)

    def query_layout(self, height: int, width: int, *, device: torch.device) -> dict:
        if height < 1 or width < 1:
            raise ValueError("native grid dimensions must be positive")
        c = torch.arange(self.config.cameras, device=device)
        t = torch.arange(self.config.future_steps, device=device)
        y = torch.arange(height, device=device)
        x = torch.arange(width, device=device)
        camera, time, row, col = torch.meshgrid(c, t, y, x, indexing="ij")
        return {"camera": camera.reshape(-1), "time": time.reshape(-1),
                "row": row.reshape(-1), "col": col.reshape(-1)}

    def forward(self, scene_kv: list[tuple[torch.Tensor, torch.Tensor]],
                anchor: torch.Tensor, *, grid: tuple[int, int]) -> torch.Tensor:
        cfg = self.config
        if len(scene_kv) != 8 or anchor.ndim != 2 or anchor.shape[0] != 3:
            raise ValueError("expected eight S+T K/V layers and mRoPE anchor [3,B]")
        height, width = grid
        layout = self.query_layout(height, width, device=anchor.device)
        count = layout["camera"].numel()
        batch = anchor.shape[1]
        for index in cfg.source_indices:
            key, value = scene_kv[index]
            if (key.shape != value.shape or key.ndim != 4 or key.shape[0] != batch
                    or key.shape[2:] != (cfg.kv_heads, cfg.head_dim)
                    or key.device != anchor.device):
                raise ValueError(f"source {index}: incompatible S+T K/V shape or device")
        coordinates = torch.stack((
            (layout["row"].float() + 0.5) * (2 / height) - 1,
            (layout["col"].float() + 0.5) * (2 / width) - 1,
        ), dim=-1)
        query = (self.base + self.camera(layout["camera"]) +
                 self.time(layout["time"]) + self.xy(coordinates))
        hidden = query.unsqueeze(0).expand(batch, -1, -1)
        # The anchor has three mRoPE axes; the Future query uses their common last
        # text position, then continues one scalar rotary position per query.
        if not torch.equal(anchor[0], anchor[1]) or not torch.equal(anchor[0], anchor[2]):
            raise ValueError("the terminal S+T mRoPE coordinates must coincide")
        steps = torch.arange(1, count + 1, device=anchor.device, dtype=anchor.dtype)
        positions = anchor[0, :, None] + steps
        for block, index in zip(self.blocks, cfg.source_indices):
            hidden = block(hidden, scene_kv[index], positions, grid)
        result = self.out_proj(self.final_norm(hidden))
        return result.reshape(batch, cfg.cameras, cfg.future_steps,
                              height, width, cfg.out_channels)


def observed_future_grid(image_grid_thw: torch.Tensor, *, merge_size: int,
                         cameras: int = 3) -> tuple[int, int]:
    """Get the target canvas from current *observed* cameras, never future GT.

    QwenDrive orders input images camera-major with the current frame last in
    each camera's history.  Future target grids must match this observable grid;
    callers check that separately after computing the prediction.
    """
    if (image_grid_thw.ndim != 2 or image_grid_thw.shape[1] != 3 or
            image_grid_thw.shape[0] % cameras or merge_size < 1):
        raise ValueError("invalid observable image_grid_thw or merge size")
    per_camera = image_grid_thw.shape[0] // cameras
    if per_camera < 1:
        raise ValueError("no observed camera frames")
    current = image_grid_thw[per_camera - 1::per_camera]
    if current.shape[0] != cameras or not torch.all(current[:, 0] == 1):
        raise ValueError("current camera grid layout differs from QwenDrive")
    rows, cols = int(current[0, 1]), int(current[0, 2])
    if (rows < merge_size or cols < merge_size or rows % merge_size or
            cols % merge_size or not torch.all(current[:, 1] == rows) or
            not torch.all(current[:, 2] == cols)):
        raise ValueError("current camera grids differ or cannot be merged")
    return rows // merge_size, cols // merge_size


def native_future_mse(prediction: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
    if prediction.shape != target.shape or prediction.ndim != 6:
        raise ValueError(f"native Future target shape {tuple(target.shape)} does not match "
                         f"prediction {tuple(prediction.shape)}")
    if not torch.isfinite(target).all():
        raise ValueError("native Future target contains nonfinite values")
    return F.mse_loss(prediction.float(), target.detach().float())
