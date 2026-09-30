"""Wrap observable data with the unmodified official scene/processor contract."""
import io
from src.utils import paths
VIEWS = ("<FRONT VIEW>", "<FRONT LEFT VIEW>", "<FRONT RIGHT VIEW>")
EXPECTED_CAM_ORDER = ("CAM_F0", "CAM_L0", "CAM_R0")

def to_driving_scene(sample: dict):
    """Wrap a dataset sample back into a ``DrivingScene`` for the official processor.

    Images are decoded from the raw bytes lazily; ``target_size`` reproduces
    exactly what the benchmark reader would set, so the QwenDriveProcessor
    applies the same smart_resize/pixel-budget path as at baseline time.
    """
    from qwen_drive.scene import CameraFrame, DrivingScene

    def decode(data: bytes):
        from PIL import Image

        return Image.open(io.BytesIO(data)).convert("RGB")

    model_input = sample["model_input"]
    frames = [
        CameraFrame(
            image=(lambda payload=b: decode(payload)),
            target_size=(w, h) if w and h else None,
        )
        for (w, h), b in zip(model_input["image_target_sizes"], model_input["images"])
    ]
    per_view = len(frames) // len(VIEWS)
    views = {
        view: frames[i * per_view : (i + 1) * per_view] for i, view in enumerate(VIEWS)
    }
    ego = model_input["ego_status"]
    return DrivingScene(
        views=views,
        history=model_input["history"],
        history_velocity=model_input["history_velocity"],
        history_acceleration=model_input["history_acceleration"],
        ego_velocity=ego[:2],
        ego_acceleration=ego[2:4],
        driving_command=model_input["driving_command"],
        nav_command=model_input["nav_command"],
        instruction_text=model_input["instruction_text"],
        token=sample["trace"]["token"],
    )

