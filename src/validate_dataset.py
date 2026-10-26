"""Optional: evaluate a verified model on a labeled local dataset split."""
import argparse
from pathlib import Path
import yaml

from common import DEFAULT_WEIGHTS, PROJECT_ROOT, choose_device, load_cardboard_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--data", default=str(PROJECT_ROOT / "dataset" / "data.yaml"))
    parser.add_argument("--split", choices=("test", "val"), default="test")
    parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()

    data_yaml = Path(args.data).expanduser().resolve()
    if not data_yaml.is_file():
        raise FileNotFoundError(
            f"Dataset YAML not found: {data_yaml}\n"
            "Copy the Roboflow YOLO-format dataset into dataset/ (including its data.yaml), "
            "or pass --data with the correct path."
        )
    config = yaml.safe_load(data_yaml.read_text(encoding="utf-8"))
    names = config.get("names", [])
    if isinstance(names, dict):
        class_names = [str(v).strip().casefold() for _, v in sorted(names.items(), key=lambda item: int(item[0]))]
    else:
        class_names = [str(v).strip().casefold() for v in names]
    if class_names != ["cardboard"]:
        raise RuntimeError(f"STOP: expected only the cardboard class; data.yaml defines {class_names}")
    if args.split not in config:
        raise RuntimeError(f"STOP: data.yaml has no '{args.split}' split. Available keys: {list(config)}")

    model = load_cardboard_model(args.weights)
    metrics = model.val(data=str(data_yaml), split=args.split, imgsz=args.imgsz, device=choose_device())
    print("Evaluation complete.")
    print(f"Precision: {metrics.box.mp:.4f}")
    print(f"Recall: {metrics.box.mr:.4f}")
    print(f"mAP50: {metrics.box.map50:.4f}")
    print(f"mAP50-95: {metrics.box.map:.4f}")


if __name__ == "__main__":
    main()
