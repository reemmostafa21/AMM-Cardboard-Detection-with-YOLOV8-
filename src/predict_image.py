"""Run cardboard detection on one still image and save the annotated result."""
import argparse
from pathlib import Path

from common import DEFAULT_WEIGHTS, PROJECT_ROOT, RUNS_DIR, choose_device, load_cardboard_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default=str(PROJECT_ROOT / "test_images" / "cardboard.jpg"))
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--show", action="store_true", help="Open an image display window if supported")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Test image not found: {source}\nAdd a JPG/PNG to test_images or pass --source with its path.")
    if not 0.0 <= args.conf <= 1.0:
        raise ValueError("--conf must be between 0 and 1")

    model = load_cardboard_model(args.weights)
    results = model.predict(
        source=str(source), conf=args.conf, imgsz=640,
        save=True, show=args.show,
        project=str(RUNS_DIR), name="image", exist_ok=True,
        device=choose_device(),
    )
    print(f"Completed image inference. Annotated output folder: {RUNS_DIR / 'image'}")
    for result in results:
        if result.boxes is None or len(result.boxes) == 0:
            print("No cardboard detections in this image. Try better lighting/angle or inspect the confidence threshold.")
        else:
            for box in result.boxes:
                class_id = int(box.cls[0].item())
                confidence = float(box.conf[0].item())
                print(f"Detected {model.names[class_id]} with confidence {confidence:.3f}")


if __name__ == "__main__":
    main()
