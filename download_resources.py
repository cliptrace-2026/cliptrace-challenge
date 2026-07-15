#!/usr/bin/env python3
"""Download CLIPTrace competition models and baseline data from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent

# Change this value to "final" when the final phase opens.
PHASE = "development"

MODEL_REPO_ID = "cliptrace-2026/cliptrace-2026-models"
DATASET_REPO_ID = "cliptrace-2026/cliptrace-baseline-data"


def download_models(phase: str, token: str | None) -> Path:
    target = BASE_DIR / "resources" / "model-repository"
    print("=" * 72)
    print(f"Downloading {phase} models from {MODEL_REPO_ID}")
    print(f"Target: {target}")
    print("=" * 72)
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        local_dir=target,
        allow_patterns=[f"models/{phase}/**", f"manifests/{phase}*", "README.md"],
        token=token,
    )
    return target / "models" / phase


def download_data(token: str | None) -> Path:
    target = BASE_DIR / "resources" / "data"
    print("\n" + "=" * 72)
    print(f"Downloading baseline data from {DATASET_REPO_ID}")
    print(f"Target: {target}")
    print("=" * 72)
    snapshot_download(
        repo_id=DATASET_REPO_ID,
        repo_type="dataset",
        local_dir=target,
        token=token,
    )
    return target / "imagenet"


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CLIPTrace competition resources")
    parser.add_argument("--phase", choices=("development", "final"), default=PHASE)
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.models_only and args.data_only:
        parser.error("--models-only and --data-only cannot be used together")

    token = os.environ.get("HF_TOKEN")
    downloaded: list[tuple[str, Path]] = []
    if not args.data_only:
        downloaded.append(("models", download_models(args.phase, token)))
    if not args.models_only:
        downloaded.append(("data", download_data(token)))

    print("\nDownload complete:")
    for kind, path in downloaded:
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

