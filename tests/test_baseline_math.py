from argparse import Namespace
from pathlib import Path

import pytest
import torch

from Baseline import main
from Baseline.utils import compute_self_cos_sim


def test_off_diagonal_similarity():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert torch.isclose(compute_self_cos_sim(features), torch.tensor(1 / 3))


def test_learning_rate_schedule_is_relative():
    parameter = torch.nn.Parameter(torch.tensor(0.0))
    optimizer = torch.optim.SGD([parameter], lr=1.0)
    args = Namespace(epochs=100, lr=0.1)

    main.adjust_learning_rate(optimizer, 0, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.1)
    main.adjust_learning_rate(optimizer, 25, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.01)
    main.adjust_learning_rate(optimizer, 75, args)
    assert optimizer.param_groups[0]["lr"] == pytest.approx(0.005)


def test_prepare_batch_matches_starter_kit_parameterization():
    clean = torch.zeros((2, 4, 4, 3), dtype=torch.float64)
    mask_logits = torch.zeros((4, 4), dtype=torch.float64)
    patch_logits = torch.zeros((4, 4, 3), dtype=torch.float64)

    poisoned, mask, patch = main.prepare_batch(
        clean,
        mask_logits,
        patch_logits,
        transform=lambda image: image,
        device=torch.device("cpu"),
    )

    assert poisoned.shape == (2, 3, 4, 4)
    assert torch.allclose(mask, torch.full_like(mask, 0.5))
    assert torch.allclose(patch, torch.full_like(patch, 127.5))
    assert torch.allclose(poisoned, torch.full_like(poisoned, 0.25))


def test_trigger_file_has_expected_starter_kit_shape():
    trigger_file = Path(main.__file__).parent / "trigger" / "trigger_clip_l.npz"
    mask, patch = main.load_trigger(trigger_file)
    assert mask.shape == (336, 336, 3)
    assert patch.shape == (336, 336, 3)


def test_default_encoder_path_uses_starter_kit_layout():
    expected = (
        main.PROJECT_ROOT
        / "cliptrace-2026-models"
        / "models"
        / "development"
        / "model_0001"
    )
    assert main.default_encoder_path("development", "model_0001") == expected


def test_auto_download_fetches_only_selected_model(tmp_path, monkeypatch):
    import huggingface_hub

    encoder_path = tmp_path / "models" / "development" / "model_0007"
    calls = []

    def fake_snapshot_download(**kwargs):
        calls.append(kwargs)
        encoder_path.mkdir(parents=True)
        (encoder_path / "config.json").write_text("{}")
        (encoder_path / "model.safetensors").write_bytes(b"test")

    monkeypatch.setattr(huggingface_hub, "snapshot_download", fake_snapshot_download)
    result = main.download_huggingface_model(
        encoder_path,
        "RobinWZQ/cliptrace-2026-models",
        "development",
        "model_0007",
        token="token",
    )

    assert result == encoder_path
    assert calls[0]["allow_patterns"] == ["models/development/model_0007/**"]
