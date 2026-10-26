"""Load the intended checkpoint and verify its class labels without running inference."""
import argparse
from pathlib import Path

from common import DEFAULT_WEIGHTS, load_cardboard_model


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", default=str(DEFAULT_WEIGHTS), help="Path to the custom cardboard .pt file")
    args = parser.parse_args()
    model = load_cardboard_model(Path(args.weights))
    print("PASS: checkpoint loaded and the single class is cardboard.")
    del model


if __name__ == "__main__":
    main()
