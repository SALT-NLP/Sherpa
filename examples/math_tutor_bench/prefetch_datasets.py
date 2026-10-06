#!/usr/bin/env python3
"""Fetch the small Hugging Face datasets used by the official task configs."""

from __future__ import annotations

import argparse
from pathlib import Path

from datasets import load_dataset

DATASETS = (
    ("gsm8k", "main", "train"),
    ("gsm8k", "main", "test"),
    ("gsm8k", "socratic", "train"),
    ("gsm8k", "socratic", "test"),
    ("eth-nlped/stepverify", "default", "train"),
)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache-dir", type=Path, required=True)
    args = parser.parse_args()
    cache_dir = args.cache_dir.resolve()
    cache_dir.mkdir(parents=True, exist_ok=True)

    for path, name, split in DATASETS:
        dataset = load_dataset(path, name, split=split, cache_dir=str(cache_dir))
        print(f"[dataset] {path}/{name}:{split} ({len(dataset)} rows)")


if __name__ == "__main__":
    main()
