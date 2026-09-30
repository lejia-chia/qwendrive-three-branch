"""Frozen QwenDrive vision targets at their native merged-patch resolution.

Future frames are supervision only.  This module never changes a training input
or the S+T VLM sequence, and never applies adaptive pooling.  The 30 frame
features are returned in camera-major, then future-time, row-major order.
"""
from __future__ import annotations

from pathlib import Path
import torch
from src.utils import paths
from qwen_drive import CameraFrame  # noqa: E402


class NativeFutureTargets:
    def __init__(self, model, *, budget: int | None = None) -> None:
        self.processor = model.processor
        self.visual = model.vlm.model.visual
        self.budget = (self.processor.config.current_image_pixels
                       if budget is None else budget)
        if self.budget < 1:
            raise ValueError("vision pixel budget must be positive")

    @torch.no_grad()
    def encode_frame(self, path: Path) -> tuple[torch.Tensor, tuple[int, int]]:
        """Return [H,W,2560] with H,W taken from the visual output grid."""
        patches, (rows, cols) = self.processor._patchify(
            CameraFrame(path), self.budget
        )
        merge = self.processor.merge_size
        if rows % merge or cols % merge:
            raise ValueError(f"patch grid {(rows, cols)} is not divisible by merge {merge}")
        height, width = rows // merge, cols // merge
        device = next(self.visual.parameters()).device
        dtype = next(self.visual.parameters()).dtype
        patches = patches.to(device=device, dtype=dtype)
        grid_thw = torch.tensor([[1, rows, cols]], device=device, dtype=torch.long)
        was_training = self.visual.training
        self.visual.eval()
        try:
            encoded = self.visual(hidden_states=patches, grid_thw=grid_thw).pooler_output
        finally:
            self.visual.train(was_training)
        channels = self.processor.config.vlm_config.text_config.hidden_size
        if encoded.shape != (height * width, channels):
            raise ValueError(f"native visual output {tuple(encoded.shape)} does not match "
                             f"grid {(height, width, channels)}")
        if not torch.isfinite(encoded).all():
            raise ValueError(f"nonfinite visual output for {path}")
        return encoded.reshape(height, width, channels).to(torch.float16).cpu(), (height, width)

    @torch.no_grad()
    def encode_scene(self, pairing: dict, *, image_root: Path) -> dict:
        """Return [3,10,H,W,2560]; reject mixed grids instead of resampling."""
        paths = pairing["relative_paths"]
        if len(paths) != 3 or any(len(column) != 10 for column in paths):
            raise ValueError("future pairing must contain 3 cameras x 10 frames")
        outputs: list[list[torch.Tensor]] = []
        expected_grid = None
        for camera_index, column in enumerate(paths):
            frames = []
            for time_index, relative_name in enumerate(column):
                relative = Path(relative_name)
                if relative.is_absolute() or ".." in relative.parts:
                    raise ValueError(f"unsafe future image path: {relative}")
                absolute = image_root / relative
                if not absolute.is_file():
                    raise FileNotFoundError(absolute)
                feature, grid = self.encode_frame(absolute)
                if expected_grid is None:
                    expected_grid = grid
                elif grid != expected_grid:
                    raise ValueError(f"native grids differ at camera {camera_index}, "
                                     f"time {time_index}: {grid} vs {expected_grid}; "
                                     "this contract never pools or silently pads targets")
                frames.append(feature)
            outputs.append(frames)
        result = torch.stack([torch.stack(frames) for frames in outputs])
        return {"token": pairing["token"], "features": result, "grid": expected_grid,
                "layout": "camera,time,row,column,channel", "budget": self.budget,
                "cameras": pairing["cameras"],
                "timestamps_us": pairing["timestamps_us"]}
