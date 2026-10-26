"""Run live cardboard detection using the default webcam (device index 0)."""
import argparse

from common import DEFAULT_WEIGHTS, RUNS_DIR, choose_device, load_cardboard_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--camera", type=int, default=0, help="Camera index: try 1 if 0 does not work")
    parser.add_argument("--conf", type=float, default=0.25)
    parser.add_argument("--save", action="store_true", help="Save annotated webcam output to a video file")
    args = parser.parse_args()

    model = load_cardboard_model(args.weights)
    print("Opening camera. Close the display or press q in its window to stop.")
    results = model.predict(
        source=args.camera, conf=args.conf, imgsz=640,
        show=True, save=args.save, stream=True,
        project=str(RUNS_DIR), name="webcam", exist_ok=True,
        device=choose_device(),
    )
    frames = 0
    try:
        for _result in results:
            frames += 1
    finally:
        close = getattr(results, "close", None)
        if callable(close):
            close()
    print(f"Webcam session ended after {frames} processed frames.")


if __name__ == "__main__":
    main()
