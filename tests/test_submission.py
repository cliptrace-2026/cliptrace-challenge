import json
import zipfile

import pytest
import torch

import create_submission


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
    payload, used = create_submission.read_and_validate_predictions(tmp_path, {"model_0001"})
    assert len(payload["predictions"]) == 1
    assert used == {"embeddings/model_0001.pt"}


def test_rejects_wrong_shape(tmp_path):
    write_prediction(tmp_path, tensor=torch.ones((1, 768), dtype=torch.float32))
    with pytest.raises(create_submission.SubmissionError, match="shape"):
        create_submission.read_and_validate_predictions(tmp_path, None)


def test_rejects_boolean_label(tmp_path):
    write_prediction(tmp_path, label=0)
    payload = json.loads((tmp_path / "predictions.json").read_text())
    payload["predictions"][0]["label"] = True
    (tmp_path / "predictions.json").write_text(json.dumps(payload))
    with pytest.raises(create_submission.SubmissionError, match="integer"):
        create_submission.read_and_validate_predictions(tmp_path, None)


def test_archive_uses_submission_root_layout(tmp_path):
    source = tmp_path / "source"
    stage = tmp_path / "stage"
    tensor = torch.nn.functional.normalize(torch.ones(768, dtype=torch.float32), dim=0)
    write_prediction(source, tensor=tensor)
    payload, used = create_submission.read_and_validate_predictions(source, None)
    create_submission.stage_submission(source, stage, payload, used)

    archive_path = tmp_path / "submission.zip"
    create_submission.create_archive(stage, archive_path)
    with zipfile.ZipFile(archive_path) as archive:
        assert set(archive.namelist()) == {"predictions.json", "embeddings/model_0001.pt"}


def test_expected_model_ids_supports_phase_directories(tmp_path):
    (tmp_path / "models" / "development" / "model_0001").mkdir(parents=True)
    (tmp_path / "models" / "development" / "model_0001" / "config.json").write_text("{}")
    assert create_submission.expected_model_ids(tmp_path / "models") == {"model_0001"}
