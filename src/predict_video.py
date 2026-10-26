"""Run detection over a recorded video using streaming inference."""
import argparse
from pathlib import Path

from common import DEFAULT_WEIGHTS, PROJECT_ROOT, RUNS_DIR, choose_device, load_cardboard_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS))
    parser.add_argument("--source", default=str(PROJECT_ROOT / "test_videos" / "cardboard_1.mp4"))
    parser.add_argument("--conf", type=float, default=0.25)
    # Show window is active only when --show is specified
    parser.add_argument("--show", action="store_true", help="Open the video display window")
    args = parser.parse_args()

    source = Path(args.source).expanduser().resolve()
    if not source.is_file():
        raise FileNotFoundError(f"Test video not found: {source}\nAdd an MP4/AVI to test_videos or pass --source with its path.")

    model = load_cardboard_model(args.weights)
    # stream=True avoids retaining every frame's result in memory for long videos.
    results = model.predict(
        source=str(source), conf=args.conf, imgsz=640,
        save=True, show=args.show, stream=True,
        project=str(RUNS_DIR), name="video", exist_ok=True,
        device=choose_device(),
    )
    count = 0
    try:
        for _result in results:
            count += 1
    finally:
        close = getattr(results, "close", None)
        if callable(close):
            close()
    print(f"Processed {count} frames. Annotated output folder: {RUNS_DIR / 'video'}")


if __name__ == "__main__":
    main()
