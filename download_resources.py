#!/usr/bin/env python3
"""Download CLIPTrace models and baseline data from Hugging Face."""

from __future__ import annotations

import argparse
import os
from pathlib import Path

from huggingface_hub import snapshot_download


BASE_DIR = Path(__file__).resolve().parent

# Change this value to "final" when the final phase opens.
PHASE = "development"

MODEL_REPO_ID = "RobinWZQ/cliptrace-2026-models"
DATASET_REPO_ID = "cliptrace-2026/cliptrace-baseline-data"
DEFAULT_MODEL_ID = "model_0001"


def is_huggingface_checkpoint(path: Path) -> bool:
    weight_files = (
        "model.safetensors",
        "model.safetensors.index.json",
        "pytorch_model.bin",
        "pytorch_model.bin.index.json",
    )
    return (path / "config.json").is_file() and any((path / name).is_file() for name in weight_files)


def download_models(
    phase: str,
    token: str | None,
    *,
    model_id: str = DEFAULT_MODEL_ID,
    all_models: bool = False,
) -> Path:
    """Download one model by default, or the complete phase on explicit request."""
    target = BASE_DIR / "cliptrace-2026-models"
    pattern = f"models/{phase}/**" if all_models else f"models/{phase}/{model_id}/**"
    description = f"all {phase} models" if all_models else f"{phase}/{model_id}"

    print("=" * 72)
    print(f"Downloading {description} from {MODEL_REPO_ID}")
    print(f"Target: {target}")
    print("=" * 72)
    snapshot_download(
        repo_id=MODEL_REPO_ID,
        repo_type="model",
        local_dir=target,
        allow_patterns=[pattern],
        token=token,
    )

    model_root = target / "models" / phase
    expected = model_root if all_models else model_root / model_id
    if all_models:
        valid = expected.is_dir() and any(
            is_huggingface_checkpoint(path) for path in expected.iterdir() if path.is_dir()
        )
    else:
        valid = is_huggingface_checkpoint(expected)
    if not valid:
        raise FileNotFoundError(f"download completed but no model checkpoint was found at {expected}")
    return expected


def download_data(token: str | None) -> Path:
    target = BASE_DIR / "Baseline" / "data"
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
    expected = target / "imagenet"
    if not expected.is_dir():
        raise FileNotFoundError(f"download completed but the ImageNet directory was not found at {expected}")
    return expected


def main() -> int:
    parser = argparse.ArgumentParser(description="Download CLIPTrace competition resources")
    parser.add_argument("--phase", choices=("development", "final"), default=PHASE)
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument(
        "--all-models",
        action="store_true",
        help="download every model in the selected phase (large; one model is downloaded by default)",
    )
    parser.add_argument("--models-only", action="store_true")
    parser.add_argument("--data-only", action="store_true")
    args = parser.parse_args()
    if args.models_only and args.data_only:
        parser.error("--models-only and --data-only cannot be used together")

    token = os.environ.get("HF_TOKEN")
    downloaded: list[tuple[str, Path]] = []
    try:
        if not args.data_only:
            downloaded.append(
                (
                    "models",
                    download_models(
                        args.phase,
                        token,
                        model_id=args.model_id,
                        all_models=args.all_models,
                    ),
                )
            )
        if not args.models_only:
            downloaded.append(("data", download_data(token)))
    except Exception as exc:
        raise RuntimeError(
            "Hugging Face download failed. Accept the repository terms and run "
            "`hf auth login`, or set an authorized HF_TOKEN."
        ) from exc

    print("\nDownload complete:")
    for kind, path in downloaded:
        print(f"  {kind}: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
