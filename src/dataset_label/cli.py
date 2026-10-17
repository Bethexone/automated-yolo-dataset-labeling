"""Command-line entry point."""

from __future__ import annotations

import argparse
from pathlib import Path
from collections.abc import Sequence


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="dataset-label")
    parser.add_argument("input_dir")
    parser.add_argument("--classes", nargs="+", required=True)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument("--test", action="store_true")
    mode.add_argument("--preview", action="store_true")
    try:
        args = parser.parse_args(argv)
    except SystemExit as error:
        return int(error.code)
    try:
        from .config import load_provider_config
        from .vision import OpenAICompatibleDetector
        from .workflow import label_directory, test_image
        config = load_provider_config(Path("config.local.toml"))
        detector = OpenAICompatibleDetector(config)
        if args.test:
            for row in test_image(Path(args.input_dir), args.classes, detector):
                print(row)
            return 0
        summary = label_directory(Path(args.input_dir), args.classes, detector, create_previews=args.preview)
        print(" ".join(f"{key}={value}" for key, value in summary.items()))
        return 0
    except (ValueError, FileNotFoundError, FileExistsError, OSError) as error:
        parser.error(str(error))
        return 2
