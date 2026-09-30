"""Verified distance-aware targets. Historical source digests are provenance only."""
from pathlib import Path
import json
import numpy as np
from src.utils.hashing import sha256_file
from src.utils.paths import DATA
FAMILY_NAMES = ("gt", "longitudinal", "lateral", "timing", "cross_scene")
STATUS_NAMES = {0:"valid",1:"near_duplicate_gt",2:"duplicate_pair",3:"invalid",9:"padding"}
RECORD_STATUS = {0:"ok",1:"gt_invalid",2:"gt_only"}
DEFAULT_Q_ARTIFACT = DATA / "q_targets/v2"
def measure(candidate: np.ndarray, candidate_mask: np.ndarray, reference: np.ndarray,
            reference_mask: np.ndarray, horizon: int):
    """ADE / FDE over the frames valid in both, inside the horizon (own code)."""
    overlap = np.logical_and(candidate_mask > 0.5, reference_mask > 0.5)
    overlap[horizon:] = False
    frames = np.where(overlap)[0]
    if frames.size == 0:
        return None
    delta = candidate[frames, :2].astype(np.float64) - reference[frames, :2].astype(np.float64)
    distances = np.sqrt(np.einsum("ij,ij->i", delta, delta))
    return float(distances.mean()), float(distances[-1]), int(frames[-1]), int(frames.size)


def score_record(trajectories, masks, counts, gt_index, horizon,
                 w_ade, w_fde, near_duplicate, *, policy="keep_all_valid", sigma=None):
    """Rebuild (status, D, q) for one record; returns lists in slot order."""
    n = int(counts)
    if not 1 <= n <= len(trajectories) or not 0 <= gt_index < n:
        raise ValueError("invalid candidate count or GT index")
    status = []
    distances = []
    for slot in range(n):
        if (not np.isfinite(trajectories[slot]).all()
                or not np.isfinite(masks[slot]).all()
                or not np.isin(masks[slot], (0, 1)).all()
                or not np.isfinite(trajectories[gt_index]).all()
                or not np.isfinite(masks[gt_index]).all()
                or not np.isin(masks[gt_index], (0, 1)).all()):
            status.append(3)
            distances.append(float("inf"))
            continue
        measured = measure(trajectories[slot], masks[slot], trajectories[gt_index],
                           masks[gt_index], horizon)
        if measured is None:
            status.append(3)
            distances.append(float("inf"))
            continue
        status.append(0)
        distances.append(w_ade * measured[0] + w_fde * measured[1])
    if not 0 <= gt_index < n or status[gt_index] != 0:
        return status, distances, [0.0] * n, "gt_invalid", {}
    if policy != "keep_all_valid":
        raise ValueError("only the current keep_all_valid target policy is supported")
    duplicate_of = {}
    live = [slot for slot in range(n) if status[slot] == 0]
    q = [0.0] * n
    if len(live) < 2:
        q[gt_index] = 1.0
        return status, distances, q, "gt_only", duplicate_of
    values = np.array([distances[slot] for slot in live], dtype=np.float64)
    if sigma is None:
        raise ValueError("explicit frozen target sigma is required")
    scale = sigma
    if not np.isfinite(scale) or scale <= 0:
        raise ValueError("sigma must be finite and positive")
    weights = np.exp(-(values - values.min()) / scale)
    weights /= weights.sum()
    for slot, weight in zip(live, weights.tolist()):
        q[slot] = weight
    return status, distances, q, "ok", duplicate_of


def load_artifact(artifact: Path) -> dict:
    manifest = json.loads((artifact / "manifest.json").read_text())
    if (manifest.get("kind") != "q_targets" or not manifest.get("complete")
            or not manifest.get("ok")):
        raise SystemExit(f"{artifact} is not a complete q_targets artifact")
    rules = json.loads((artifact / "rules.json").read_text())
    if manifest.get("rules_sha256") != sha256_file(artifact / "rules.json"):
        raise SystemExit("rules.json does not match its recorded digest")
    if manifest["sigma"] != manifest["sigma_source"]["quantiles"]["median"]:
        raise SystemExit("manifest sigma is not the recorded train median")
    if rules["exclusions"].get("policy") != "keep_all_valid":
        raise ValueError("only current keep_all_valid artifacts are supported")
    splits = {}
    for split, info in manifest["splits"].items():
        arrays_path, meta_path = artifact / info["file"], artifact / info["metadata"]
        if sha256_file(arrays_path) != info["sha256"]:
            raise SystemExit(f"{arrays_path} digest differs from the manifest")
        if sha256_file(meta_path) != info["metadata_sha256"]:
            raise SystemExit(f"{meta_path} digest differs from the manifest")
        with np.load(arrays_path, allow_pickle=False) as blob:
            arrays = {name: blob[name] for name in blob.files}
        n = info["rows"]
        expected_shapes = {"trajectories": (n, 7, 50, 3), "masks": (n, 7, 50),
                           "counts": (n,), "gt_indices": (n,), "record_status": (n,),
                           "tokens": (n,)}
        for name in ("families", "ade4", "fde4", "fde4_step", "d", "q", "valid", "status"):
            expected_shapes[name] = (n, 7)
        if rules["exclusions"].get("policy") == "keep_all_valid":
            expected_shapes.update(trainable=(n,), near_gt=(n, 7))
        for name, shape in expected_shapes.items():
            if name not in arrays or arrays[name].shape != shape:
                raise SystemExit(f"{split}: {name} has incompatible shape")
        for name in ("counts", "gt_indices", "families", "record_status", "status", "fde4_step", "horizon_steps"):
            if not np.issubdtype(arrays[name].dtype, np.integer):
                raise SystemExit(f"{split}: {name} must have integer dtype")
        if not np.isfinite(manifest["sigma"]) or manifest["sigma"] <= 0:
            raise SystemExit("sigma must be finite and positive")
        if (len(set(arrays["tokens"].tolist())) != n
                or float(arrays["sigma"]) != manifest["sigma"]
                or int(arrays["horizon_steps"]) != rules["horizon"]["steps"]):
            raise SystemExit(f"{split}: duplicate tokens or sigma/horizon mismatch")
        rows = [json.loads(line) for line in meta_path.open() if line.strip()]
        if len(rows) != info["rows"] or arrays["tokens"].tolist() != [r["token"] for r in rows]:
            raise SystemExit(f"{split}: metadata rows differ from the arrays")
        splits[split] = {"arrays": arrays, "rows": rows, "info": info}
    return {"manifest": manifest, "rules": rules, "splits": splits, "root": artifact}


def validate_record(arrays, row: int, metadata: dict, rules: dict, sigma: float) -> list[str]:
    """Validate stored fields against independent recomputation; used by negative tests."""
    errors = []
    def require(condition, message):
        if not condition:
            errors.append(f"row {row}: {message}")
    for name in ("counts", "gt_indices", "families", "record_status", "status", "fde4_step", "horizon_steps"):
        if not np.issubdtype(arrays[name].dtype, np.integer):
            errors.append(f"row {row}: {name} must have integer dtype")
    if errors:
        return errors
    count, gt = int(arrays["counts"][row]), int(arrays["gt_indices"][row])
    slots = arrays["q"].shape[1]
    if not 1 <= count <= slots or not 0 <= gt < count:
        return [f"row {row}: invalid count/GT index"]
    if not np.isin(arrays["families"][row, :count], np.arange(len(FAMILY_NAMES))).all():
        return [f"row {row}: occupied candidate family out of range"]
    horizon = int(arrays["horizon_steps"])
    if not 1 <= horizon <= arrays["trajectories"].shape[2]:
        return [f"row {row}: invalid horizon"]
    measured_status, distance, q, record_status, _ = score_record(
        arrays["trajectories"][row], arrays["masks"][row], count, gt, horizon,
        rules["distance"]["w_ade"], rules["distance"]["w_fde"],
        rules["exclusions"]["near_duplicate_m"],
        policy=rules["exclusions"].get("policy"), sigma=sigma)
    require(RECORD_STATUS.get(int(arrays["record_status"][row])) == record_status,
            "record_status differs from recompute")
    require(metadata.get("record_status") == record_status, "metadata record_status differs")
    expected_valid = np.zeros(slots, dtype=np.uint8)
    expected_valid[:count] = np.asarray(measured_status) == 0
    require(np.array_equal(arrays["valid"][row], expected_valid), "valid differs from recompute")
    require(np.array_equal(arrays["status"][row, :count], measured_status), "status differs")
    require(np.isfinite(arrays["q"][row]).all() and (arrays["q"][row] >= 0).all(), "invalid q")
    require(np.allclose(arrays["q"][row, :count], q, rtol=0, atol=1e-12), "q differs")
    require((arrays["q"][row, expected_valid == 0] == 0).all(), "q outside valid support")
    require(np.array_equal(arrays["d"][row, :count], distance), "D differs")
    require((arrays["status"][row, count:] == 9).all()
            and (arrays["families"][row, count:] == -1).all()
            and (arrays["masks"][row, count:] == 0).all()
            and np.isfinite(arrays["trajectories"][row, count:]).all()
            and (arrays["trajectories"][row, count:] == 0).all(), "padding differs")
    require(np.count_nonzero(arrays["families"][row, :count] == 0) == 1
            and arrays["families"][row, gt] == 0, "GT family/index differs")
    require(metadata.get("count") == count and metadata.get("gt_index") == gt,
            "metadata count/GT index differs")
    require(metadata.get("sigma") == sigma and metadata.get("horizon_steps") == horizon,
            "metadata sigma/horizon differs")
    if rules["exclusions"].get("policy") == "keep_all_valid":
        require(int(arrays["trainable"][row]) == int(record_status == "ok"), "trainable differs")
        require(metadata.get("trainable") == (record_status == "ok"), "metadata trainable differs")
    candidates = metadata.get("candidates", [])
    require(len(candidates) == count, "metadata candidate count differs")
    for slot, candidate in enumerate(candidates[:count]):
        require(candidate.get("slot") == slot
                and candidate.get("family") == FAMILY_NAMES[int(arrays["families"][row, slot])],
                f"candidate {slot} identity differs")
        require(candidate.get("valid") == bool(expected_valid[slot]), f"candidate {slot} valid differs")
        require(candidate.get("status") == STATUS_NAMES[measured_status[slot]], f"candidate {slot} status differs")
        require(candidate.get("q") == float(arrays["q"][row, slot]), f"candidate {slot} q differs")
        expected_d = distance[slot] if np.isfinite(distance[slot]) else None
        require(candidate.get("d_m") == expected_d, f"candidate {slot} D differs")
        measured = measure(arrays["trajectories"][row, slot], arrays["masks"][row, slot],
                           arrays["trajectories"][row, gt], arrays["masks"][row, gt], horizon)
        if measured is not None and np.isfinite(measured[:2]).all():
            ade, fde, frame, frames = measured
            require(arrays["ade4"][row, slot] == ade and arrays["fde4"][row, slot] == fde
                    and arrays["fde4_step"][row, slot] == frame, f"candidate {slot} metrics differ")
            require(candidate.get("ade4_m") == ade and candidate.get("fde4_m") == fde
                    and candidate.get("fde4_step") == frame
                    and candidate.get("fde4_at_horizon") == (frame == horizon-1)
                    and candidate.get("valid_frames") == frames, f"candidate {slot} metadata metrics differ")
        if rules["exclusions"].get("policy") == "keep_all_valid":
            near = slot != gt and measured_status[slot] == 0 and distance[slot] < rules["exclusions"]["near_duplicate_m"]
            require(bool(arrays["near_gt"][row, slot]) == near and candidate.get("near_gt") == near,
                    f"candidate {slot} near_gt differs")
    return errors

def load_q_artifact(root: Path, split: str) -> dict:
    root = Path(root).resolve()
    artifact = load_artifact(root)
    entry = artifact["splits"][split]
    for row, metadata in enumerate(entry["rows"]):
        errors = validate_record(entry["arrays"], row, metadata, artifact["rules"], artifact["manifest"]["sigma"])
        if errors:
            raise SystemExit(f"{root}/{split}: invalid q record: {errors[:3]}")
    return {**entry, "manifest": artifact["manifest"], "rules": artifact["rules"],
            "root": str(root), "manifest_sha256": sha256_file(root / "manifest.json"),
            "rules_sha256": artifact["manifest"]["rules_sha256"],
            "sigma": artifact["manifest"]["sigma"]}


def target_binding(entry: dict) -> dict:
    return {key: entry[key] for key in ("root", "manifest_sha256", "rules_sha256", "sigma")} | {
        "policy": entry["rules"]["exclusions"]["policy"]}

