import argparse
import json
import math
import shutil
import tempfile
import zipfile
from pathlib import Path, PurePosixPath


PROJECTION_DIM = 768


class SubmissionError(ValueError):
    pass


def safe_relative_path(value):
    path = PurePosixPath(value)
    if path.is_absolute() or not path.parts or ".." in path.parts:
        raise SubmissionError(f"illegal relative path: {value!r}")
    return path


def expected_model_ids(models_dir):
    if models_dir is None or not models_dir.is_dir():
        return None
    values = {
        path.parent.name
        for path in models_dir.rglob("config.json")
        if path.is_file()
    }
    return values or None


def load_embedding(path):
    import torch

    try:
        return torch.load(path, map_location="cpu", weights_only=True)
    except TypeError:
        return torch.load(path, map_location="cpu")


def read_and_validate_predictions(submission_dir, expected_ids):
    import torch

    predictions_path = submission_dir / "predictions.json"
    if not predictions_path.is_file():
        raise SubmissionError("submission/predictions.json is missing")

    try:
        payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise SubmissionError(f"invalid predictions.json: {exc}") from exc

    if payload.get("version") != "1.0" or not isinstance(payload.get("predictions"), list):
        raise SubmissionError("predictions.json must contain version='1.0' and a predictions list")

    seen = set()
    used_embeddings = set()
    for index, item in enumerate(payload["predictions"]):
        if not isinstance(item, dict) or set(item) != {"model_id", "label", "embedding_file"}:
            raise SubmissionError(f"prediction {index} must contain exactly model_id, label, embedding_file")

        model_id = item["model_id"]
        label = item["label"]
        embedding_file = item["embedding_file"]
        if not isinstance(model_id, str) or not model_id or model_id in seen:
            raise SubmissionError(f"invalid or duplicate model_id at prediction {index}")
        seen.add(model_id)

        if type(label) is not int or label not in (0, 1):
            raise SubmissionError(f"label for {model_id} must be integer 0 or 1")

        if label == 0:
            if embedding_file is not None:
                raise SubmissionError(f"embedding_file for normal prediction {model_id} must be null")
            continue

        if not isinstance(embedding_file, str) or not embedding_file.endswith(".pt"):
            raise SubmissionError(f"backdoor prediction {model_id} requires a .pt embedding_file")
        relative = safe_relative_path(embedding_file)
        if relative.parts[0] != "embeddings":
            raise SubmissionError(f"embedding for {model_id} must be under embeddings/")

        embedding_path = submission_dir.joinpath(*relative.parts)
        if not embedding_path.is_file():
            raise SubmissionError(f"missing embedding file: {embedding_file}")
        tensor = load_embedding(embedding_path)
        if not torch.is_tensor(tensor) or tuple(tensor.shape) != (PROJECTION_DIM,):
            raise SubmissionError(f"{embedding_file} must contain one tensor with shape [{PROJECTION_DIM}]")
        if tensor.dtype != torch.float32 or tensor.device.type != "cpu":
            raise SubmissionError(f"{embedding_file} must be a CPU torch.float32 tensor")
        if not torch.isfinite(tensor).all():
            raise SubmissionError(f"{embedding_file} contains NaN or infinity")
        if not math.isclose(float(tensor.norm()), 1.0, abs_tol=1e-4):
            raise SubmissionError(f"{embedding_file} is not L2-normalized")
        used_embeddings.add(relative.as_posix())

    if expected_ids is not None and seen != expected_ids:
        missing = sorted(expected_ids - seen)
        extra = sorted(seen - expected_ids)
        raise SubmissionError(f"model-id mismatch: missing={missing}, extra={extra}")

    return payload, used_embeddings


def stage_submission(submission_dir, stage_dir, payload, used_embeddings):
    (stage_dir / "embeddings").mkdir(parents=True, exist_ok=True)
    (stage_dir / "predictions.json").write_text(
        json.dumps(payload, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    for embedding_file in sorted(used_embeddings):
        relative = safe_relative_path(embedding_file)
        source = submission_dir.joinpath(*relative.parts)
        target = stage_dir.joinpath(*relative.parts)
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(source, target)


def create_archive(stage_dir, output):
    output.parent.mkdir(parents=True, exist_ok=True)
    included = 0
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for path in sorted(stage_dir.rglob("*")):
            if not path.is_file():
                continue
            relative = path.relative_to(stage_dir).as_posix()
            safe_relative_path(relative)
            archive.write(path, relative)
            included += 1
    if not zipfile.is_zipfile(output):
        raise SubmissionError("failed to create a readable ZIP archive")
    return included


def main():
    parser = argparse.ArgumentParser(description="Validate and package a CLIPTrace submission")
    parser.add_argument("--submission-dir", type=Path, default=Path("submission"))
    parser.add_argument("--output", type=Path, default=Path("submission.zip"))
    parser.add_argument("--models-dir", type=Path, default=Path("cliptrace-2026-models/models"))
    parser.add_argument("--skip-model-id-check", action="store_true")
    args = parser.parse_args()

    expected_ids = None if args.skip_model_id_check else expected_model_ids(args.models_dir)
    payload, used_embeddings = read_and_validate_predictions(args.submission_dir, expected_ids)

    with tempfile.TemporaryDirectory(prefix="cliptrace-submission-") as temp_dir:
        stage_dir = Path(temp_dir) / "submission"
        stage_submission(args.submission_dir, stage_dir, payload, used_embeddings)
        file_count = create_archive(stage_dir, args.output)

    print(
        f"Created {args.output} with {file_count} files "
        f"({len(payload['predictions'])} predictions, {len(used_embeddings)} embeddings)."
    )


if __name__ == "__main__":
    main()
