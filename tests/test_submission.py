import importlib.util
import json
import zipfile
from pathlib import Path

import pytest
import torch


ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location("create_submission", ROOT / "create_submission.py")
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def write_prediction(root, label=1, tensor=None):
    (root / "embeddings").mkdir(parents=True)
    embedding_file = "embeddings/model_0001.pt" if label else None
    (root / "predictions.json").write_text(
        json.dumps(
            {
                "version": "1.0",
                "predictions": [
                    {"model_id": "model_0001", "label": label, "embedding_file": embedding_file}
                ],
            }
        ),
        encoding="utf-8",
    )
    if tensor is not None:
        torch.save(tensor, root / "embeddings" / "model_0001.pt")


def test_valid_submission(tmp_path):
    tensor = torch.nn.functional.normalize(torch.ones(768, dtype=torch.float32), dim=0)
    write_prediction(tmp_path, tensor=tensor)
    assert MODULE.validate_submission_directory(tmp_path, {"model_0001"}) == {
        "predictions": 1,
        "embeddings": 1,
    }


def test_rejects_wrong_shape(tmp_path):
    write_prediction(tmp_path, tensor=torch.ones((1, 768), dtype=torch.float32))
    with pytest.raises(MODULE.SubmissionError, match="shape"):
        MODULE.validate_submission_directory(tmp_path)


def test_rejects_boolean_label(tmp_path):
    write_prediction(tmp_path, label=0)
    payload = json.loads((tmp_path / "predictions.json").read_text())
    payload["predictions"][0]["label"] = True
    (tmp_path / "predictions.json").write_text(json.dumps(payload))
    with pytest.raises(MODULE.SubmissionError, match="integer"):
        MODULE.validate_submission_directory(tmp_path)


def test_archive_uses_submission_root_layout(tmp_path):
    tensor = torch.nn.functional.normalize(torch.ones(768, dtype=torch.float32), dim=0)
    write_prediction(tmp_path, tensor=tensor)
    archive_path = tmp_path.parent / "submission.zip"
    MODULE.create_archive(tmp_path, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"predictions.json", "embeddings/model_0001.pt"}
