"""Shared path/model helpers for the AMM cardboard test project."""
from pathlib import Path
from typing import Any

from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_WEIGHTS = PROJECT_ROOT / "weights" / "yolov8n_best.pt"
RUNS_DIR = PROJECT_ROOT / "runs" / "local_test"


def load_cardboard_model(weights_path: str | Path = DEFAULT_WEIGHTS) -> YOLO:
    """Load a custom one-class cardboard model and reject the default COCO model."""
    path = Path(weights_path).expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(
            f"Model checkpoint was not found: {path}\n"
            "Copy your verified custom *_best.pt file into the project's weights folder, "
            "or supply --weights with its full path. Do not use plain yolov8n.pt here."
        )

    model = YOLO(str(path))
    raw_names: Any = model.names
    if isinstance(raw_names, dict):
        names = [str(value).strip().casefold() for _, value in sorted(raw_names.items(), key=lambda item: int(item[0]))]
    else:
        names = [str(value).strip().casefold() for value in raw_names]

    print(f"Loaded checkpoint: {path}")
    print(f"Model classes: {names}")
    if names != ["cardboard"]:
        raise RuntimeError(
            "SAFETY CHECK FAILED: Expected exactly ['cardboard'] but this checkpoint has "
            f"{names}. It may be the original COCO model or a checkpoint from the wrong run. "
            "Do not use it for the project's cardboard test until you verify the weights."
        )
    return model


def choose_device() -> int | str:
    """Prefer CUDA when PyTorch can access it; otherwise use CPU."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"Inference device: {torch.cuda.get_device_name(0)}")
            return 0
    except Exception as exc:
        print(f"CUDA detection failed; falling back to CPU ({exc})")
    print("Inference device: CPU (may be slower)")
    return "cpu"
