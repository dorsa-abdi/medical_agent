import argparse
from pathlib import Path

from src.delphi.training import train


def main():
    parser = argparse.ArgumentParser(description="Train a research-only Delphi-compatible checkpoint")
    parser.add_argument("--data", type=Path, required=True, help="JSONL trajectories")
    parser.add_argument("--vocabulary", type=Path, default=Path("data/delphi/vocabulary.json"))
    parser.add_argument("--output", type=Path, default=Path("checkpoints/delphi.pt"))
    parser.add_argument("--epochs", type=int, default=10)
    args = parser.parse_args()
    train(args.data, args.vocabulary, args.output, args.epochs)


if __name__ == "__main__":
    main()
