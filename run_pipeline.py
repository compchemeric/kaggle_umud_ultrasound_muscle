"""Run the configured ultrasound-processing experiment."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from src.orchestrator import run_from_config


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--config",
        type=Path,
        default=Path(__file__).resolve().parent / "pipeline_config.yaml",
    )
    args = parser.parse_args()
    logging.basicConfig(
        level=logging.INFO,
        format="%(levelname)s: %(message)s",
    )
    records = run_from_config(args.config.resolve())
    print(f"Processed {len(records)} images.")


if __name__ == "__main__":
    main()
