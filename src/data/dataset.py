"""Native scene and future-frame dataset. No pooled latent stores."""
from __future__ import annotations
import os
import json
import hashlib
from pathlib import Path
import numpy as np
import torch
from src.utils.paths import ROOT
from src.utils.hashing import sha256_file
from src.data.scene import EXPECTED_CAM_ORDER,VIEWS,to_driving_scene

class NativeSceneReader:
    """Sequence records for one split: prompt inputs + GT trajectory, no targets.

    Deliberately separate from ``PairedNavtrainDataset``, whose ``__getitem__`` also
    reads the pooled latent for the same row — that read is exactly what the new
    chain must not do.
    """

    def __init__(self, index_path: Path, *, records: dict[str, dict] | None = None):
        self.index_path = Path(index_path)
        self.rows = [json.loads(line) for line in self.index_path.open()]
        self.lookup = {row["token"]: position for position, row in enumerate(self.rows)}
        if records is not None:
            self._records = records
        else:
            self._records = {row["token"]: row for row in self.rows}
        self._scene_handles: dict[str, object] = {}
        self._pid = os.getpid()

    def __getstate__(self):
        """Spawn workers must open their own scene files."""
        state = self.__dict__.copy()
        state["_scene_handles"] = {}
        state["_pid"] = None
        return state

    def close(self) -> None:
        for handle in self._scene_handles.values():
            handle.close()
        self._scene_handles.clear()

    def _record(self, token: str) -> dict:
        row = self._records[token]
        if self._pid != os.getpid():
            # Forked workers inherit file descriptors and their shared offset.
            # A worker must discard those descriptors before its first seek.
            self.close()
            self._pid = os.getpid()
        handle = self._scene_handles.get(row["scene_path"])
        if handle is None:
            handle = open(ROOT / row["scene_path"], "rb")
            self._scene_handles[row["scene_path"]] = handle
        handle.seek(row["scene_byte_offset"])
        blob = handle.read(row["scene_byte_length"])
        if len(blob) != row["scene_byte_length"]:
            raise ValueError(f"{token}: short scene record")
        expected = row.get("record_sha256")
        if expected is not None and hashlib.sha256(blob).hexdigest() != expected:
            raise ValueError(f"{token}: scene record digest differs from native index")
        record, _ = json.JSONDecoder().raw_decode(blob.decode("utf-8"))
        return record

    def sample(self, token: str) -> dict:
        row = self._records[token]
        record = self._record(token)
        content = record["messages"][0]["content"]
        image_items = [item for item in content if "image" in item]
        if len(image_items) != 12:
            raise ValueError(f"{token}: expected 12 prompt images, got {len(image_items)}")
        relative_paths = [item["image"] for item in image_items]
        for camera_index, camera in enumerate(EXPECTED_CAM_ORDER):
            for path in relative_paths[camera_index * 4:(camera_index + 1) * 4]:
                if path.split("/")[1] != camera:
                    raise ValueError(f"{token}: camera grouping broken for {path}")
        instruction = [item["text"] for item in content if "text" in item][-1]
        trajectory = record["trajectory"]
        return {
            "token": token, "log_name": row["log_name"],
            "model_input": {
                "images": [item["image"] for item in image_items],
                "image_target_sizes": [(item.get("resized_width"), item.get("resized_height"))
                                       for item in image_items],
                "view_order": list(VIEWS),
                "instruction_text": instruction,
                "nav_command": int(trajectory["nav_command"]),
                "driving_command": np.asarray(
                    trajectory["ego_status"]["driving_command"], dtype=np.float32),
                "history": np.asarray(trajectory["hist_traj_10hz"], dtype=np.float32),
                "history_velocity": np.asarray(trajectory["hist_vel_10hz"], dtype=np.float32),
                "history_acceleration": np.asarray(trajectory["hist_acc_10hz"], dtype=np.float32),
                "ego_status": np.concatenate([
                    np.asarray(trajectory["ego_status"]["ego_velocity"], dtype=np.float32),
                    np.asarray(trajectory["ego_status"]["ego_acceleration"], dtype=np.float32),
                    np.asarray(trajectory["ego_status"]["driving_command"], dtype=np.float32)]),
            },
            "trajectory": {
                "future_traj_10hz": np.asarray(trajectory["future_traj_10hz"], dtype=np.float32),
                "future_valid_mask_10hz": np.asarray(trajectory["future_valid_mask_10hz"],
                                                     dtype=np.float32),
                "hist_traj_10hz": np.asarray(trajectory["hist_traj_10hz"], dtype=np.float32),
                "nav_command": int(trajectory["nav_command"]),
            },
        }


class NativeNavtrainDataset:
    """Observed scene, future image paths and trajectory supervision, token-aligned."""

    def __init__(self, index_dir: Path, processor, *, image_root: Path | None = None,
                 tokens: list[str] | None = None, split: str | None = None,
                 
                 expect_version: str | None = None):
        self.index_dir = Path(index_dir)
        manifest_path = self.index_dir / "manifest.json"
        self.manifest = json.loads(manifest_path.read_text())
        if not self.manifest.get("complete"):
            raise SystemExit(f"{manifest_path} is not complete")
        if expect_version and self.manifest.get("version") != expect_version:
            raise SystemExit(f"native index version {self.manifest.get('version')} "
                             f"!= expected {expect_version}")
        # the index is split-agnostic: every row carries its own split
        self.processor = processor
        self._rows: dict[str, list[dict]] = {}
        self._readers: dict[str, NativeSceneReader] = {}
        for index_split in self.manifest["splits"]:
            info = self.manifest['splits'][index_split]
            index_path = self.index_dir / info['file']
            if sha256_file(index_path) != info['sha256']:
                raise ValueError(f"{index_split}: native index digest mismatch")
            rows = [json.loads(line) for line in index_path.open()]
            if len(rows) != info['rows']:
                raise ValueError(f"{index_split}: native row count mismatch")
            self._rows[index_split] = rows
            self._readers[index_split] = NativeSceneReader(
                self.index_dir / f"{index_split}.jsonl",
                records={row["token"]: row for row in rows})
        if image_root is not None:
            self.image_root = Path(image_root)
        else:
            roots = {info.get("image_root") for info in self.manifest["splits"].values()}
            roots.discard(None)
            if len(roots) == 1:
                self.image_root = Path(next(iter(roots)))
            elif not roots:
                raise ValueError("native manifest must specify image_root")
            else:
                raise SystemExit("the native index records more than one image_root; "
                                 f"pass image_root explicitly (found {roots})")
        self.lookup: dict[str, tuple[str, int]] = {}
        for index_split, rows in self._rows.items():
            for position, row in enumerate(rows):
                if row["token"] in self.lookup:
                    raise SystemExit(f"token {row['token']} appears in more than one split")
                self.lookup[row["token"]] = (index_split, position)
        if split is not None and split not in self._rows:
            raise ValueError(f"unknown native split {split!r}")
        if tokens is None and split is None:
            raise ValueError("pass split='train' or split='val' when tokens is omitted")
        self.tokens = (list(tokens) if tokens is not None else
                       [row["token"] for row in self._rows[split]])
        missing = [token for token in self.tokens if token not in self.lookup]
        if missing:
            raise SystemExit(f"{len(missing)} tokens are not in the native index "
                             f"(first {missing[0]})")
        if split is not None:
            wrong = [token for token in self.tokens if self.lookup[token][0] != split]
            if wrong:
                raise SystemExit(f"{wrong[0]} belongs to {self.lookup[wrong[0]][0]}, "
                                 f"not the requested {split} split")

    def __len__(self) -> int:
        return len(self.tokens)

    def sample_at(self, token: str) -> dict:
        """Poisonable seam for the leak test, exactly like the reasoning dataset."""
        split, position = self.lookup[token]
        row = self._rows[split][position]
        scene = self._readers[split].sample(token)
        return {"row": row, "scene": scene}

    def __getitem__(self, position: int) -> dict:
        token = self.tokens[position]
        bound = self.sample_at(token)
        row, scene = bound["row"], bound["scene"]
        # image bytes are read here (not in the reader) so a poisoned seam can swap them
        images = [(self.image_root / relative).read_bytes()
                  for relative in scene["model_input"]["images"]]
        model_input = dict(scene["model_input"], images=images)
        inputs = self.processor(
            to_driving_scene({"model_input": model_input,
                              "trace": {"token": token, "log_name": row["log_name"]}}),
            with_reasoning=True, device="cpu")
        frames = row["future_frames"]
        if len(frames["relative_paths"]) != len(frames["cameras"]):
            raise SystemExit(f"{token}: future_frames is not camera-major")
        return {
            "inputs": inputs,
            "future_frames": frames,
            "trajectory": scene["trajectory"],
            "token": token,
            "log_name": row["log_name"],
            "split": self.lookup[token][0],
            "record_sha256": row["record_sha256"],
        }

