from pathlib import Path

import download_resources


def test_download_one_model_uses_correct_repository_and_pattern(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(download_resources, "BASE_DIR", tmp_path)

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        model_dir = (
            Path(kwargs["local_dir"])
            / "models"
            / "development"
            / "model_0007"
        )
        model_dir.mkdir(parents=True)
        (model_dir / "config.json").write_text("{}")
        (model_dir / "model.safetensors").write_bytes(b"test")

    monkeypatch.setattr(download_resources, "snapshot_download", fake_snapshot_download)
    result = download_resources.download_models(
        "development",
        "token",
        model_id="model_0007",
    )

    assert result.name == "model_0007"
    assert calls == [
        {
            "repo_id": "RobinWZQ/cliptrace-2026-models",
            "repo_type": "model",
            "local_dir": tmp_path / "cliptrace-2026-models",
            "allow_patterns": ["models/development/model_0007/**"],
            "token": "token",
        }
    ]


def test_download_data_uses_baseline_data_layout(tmp_path, monkeypatch):
    monkeypatch.setattr(download_resources, "BASE_DIR", tmp_path)

    def fake_snapshot_download(**kwargs):
        assert kwargs["repo_id"] == "cliptrace-2026/cliptrace-baseline-data"
        assert kwargs["repo_type"] == "dataset"
        (Path(kwargs["local_dir"]) / "imagenet" / "val").mkdir(parents=True)

    monkeypatch.setattr(download_resources, "snapshot_download", fake_snapshot_download)
    result = download_resources.download_data(None)
    assert result == tmp_path / "Baseline" / "data" / "imagenet"
