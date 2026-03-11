import argparse

from infominrev.config import load_config
from infominrev.engine import Trainer


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True, help="Path to config JSON")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(args.config)
    Trainer(config).train()


if __name__ == "__main__":
    main()
