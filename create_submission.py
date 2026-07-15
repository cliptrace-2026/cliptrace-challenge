#!/usr/bin/env python3
"""Validate submission/ and create submission.zip."""

from __future__ import annotations

import argparse
import json
import math
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PROJECTION_DIM = 768


class SubmissionError(ValueError):
    pass


def safe_relative_path(value: str) -> PurePosixPath:
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SubmissionError(f"Illegal relative path: {value!r}")
    return path


def discover_expected_model_ids(models_dir: Path | None) -> set[str] | None:
    if models_dir is None or not models_dir.is_dir():
        return None
    values = {path.name for path in models_dir.iterdir() if path.is_dir() and (path / "config.json").is_file()}
    return values or None


def validate_submission_directory(root: Path, expected_model_ids: set[str] | None = None) -> dict:
    import torch

    predictions_path = root / "predictions.json"
    if not predictions_path.is_file():
        raise SubmissionError("submission/predictions.json is missing")
    try:
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"Invalid predictions.json: {exc}") from exc
    if payload.get("version") != "1.0" or not isinstance(payload.get("predictions"), list):
        raise SubmissionError("predictions.json must contain version='1.0' and a predictions list")

    seen: set[str] = set()
    used_embeddings: set[str] = set()
    for index, item in enumerate(payload["predictions"]):
        if not isinstance(item, dict):
            raise SubmissionError(f"Prediction {index} must be a JSON object")
        if set(item) != {"model_id", "label", "embedding_file"}:
            raise SubmissionError(f"Prediction {index} has missing or unexpected fields")
        model_id = item["model_id"]
        label = item["label"]
        embedding_file = item["embedding_file"]
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise SubmissionError(f"Invalid or duplicate model_id at prediction {index}")
        seen.add(model_id)
        if type(label) is not int or label not in (0, 1):
            raise SubmissionError(f"label for {model_id} must be integer 0 or 1")

        if label == 0:
            if embedding_file is not None:
                raise SubmissionError(f"embedding_file for normal prediction {model_id} must be null")
            continue
        if not isinstance(embedding_file, str) or not embedding_file.endswith(".pt"):
            raise SubmissionError(f"Backdoor prediction {model_id} requires a .pt embedding_file")
        relative = safe_relative_path(embedding_file)
        if relative.parts[0] != "embeddings":
            raise SubmissionError(f"Embedding for {model_id} must be stored under embeddings/")
        file_path = root.joinpath(*relative.parts)
        if not file_path.is_file():
            raise SubmissionError(f"Missing embedding file: {embedding_file}")
        tensor = torch.load(file_path, map_location="cpu", weights_only=True)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != (PROJECTION_DIM,):
            raise SubmissionError(f"{embedding_file} must contain one tensor with shape [{PROJECTION_DIM}]")
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise SubmissionError(f"{embedding_file} must be a CPU torch.float32 tensor")
        if not torch.isfinite(tensor).all():
            raise SubmissionError(f"{embedding_file} contains NaN or infinity")
        if not math.isclose(float(tensor.norm()), 1.0, abs_tol=1e-4):
            raise SubmissionError(f"{embedding_file} is not L2-normalized")
        used_embeddings.add(relative.as_posix())

    if expected_model_ids is not None and seen != expected_model_ids:
        missing = sorted(expected_model_ids - seen)
        extra = sorted(seen - expected_model_ids)
        raise SubmissionError(f"Model-id mismatch: missing={missing}, extra={extra}")

    existing_embeddings = {
        path.relative_to(root).as_posix()
        for path in (root / "embeddings").glob("*.pt")
        if path.is_file()
    }
    if existing_embeddings != used_embeddings:
        unused = sorted(existing_embeddings - used_embeddings)
        missing = sorted(used_embeddings - existing_embeddings)
        raise SubmissionError(f"Embedding file mismatch: unused={unused}, missing={missing}")
    return {"predictions": len(seen), "embeddings": len(used_embeddings)}


def create_archive(submission_dir: Path, output: Path) -> int:
    included = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(submission_dir.rglob("*")):
            if not path.is_file() or path.name == ".gitkeep" or "__pycache__" in path.parts:
                continue
            relative = path.relative_to(submission_dir).as_posix()
            safe_relative_path(relative)
            archive.write(path, relative)
            included += 1
    return included


def main() -> int:
    root = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Validate and package a CLIPTrace submission")
    parser.add_argument("--submission-dir", type=Path, default=root / "submission")
    parser.add_argument("--output", type=Path, default=root / "submission.zip")
    parser.add_argument("--phase", choices=("development", "final"), default="development")
    parser.add_argument("--models-dir", type=Path)
    parser.add_argument("--skip-model-id-check", action="store_true")
    args = parser.parse_args()

    models_dir = args.models_dir or root / "resources" / "model-repository" / "models" / args.phase
    expected_ids = None if args.skip_model_id_check else discover_expected_model_ids(models_dir)
    result = validate_submission_directory(args.submission_dir, expected_ids)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    file_count = create_archive(args.submission_dir, args.output)
    if not zipfile.is_zipfile(args.output):
        raise SubmissionError("Failed to create a readable ZIP archive")
    print(
        f"Created {args.output} with {file_count} files "
        f"({result['predictions']} predictions, {result['embeddings']} embeddings)."
    )
    if expected_ids is None and not args.skip_model_id_check:
        print("Warning: model directory was not found; model-id completeness was not checked.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

