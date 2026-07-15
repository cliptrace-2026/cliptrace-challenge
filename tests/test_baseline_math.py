import importlib.util
from pathlib import Path

import torch


ROOT = Path(__file__).resolve().parents[1]
SCRIPT = ROOT / "detection-recovery-track" / "scripts" / "baseline_decree.py"
SPEC = importlib.util.spec_from_file_location("baseline_decree", SCRIPT)
MODULE = importlib.util.module_from_spec(SPEC)
assert SPEC.loader is not None
SPEC.loader.exec_module(MODULE)


def test_off_diagonal_similarity():
    features = torch.tensor([[1.0, 0.0], [1.0, 0.0], [0.0, 1.0]])
    assert torch.isclose(MODULE.off_diagonal_similarity(features), torch.tensor(1 / 3))


def test_learning_rate_schedule_is_relative():
    assert MODULE.learning_rate_factor(0, 100) == 1.0
    assert MODULE.learning_rate_factor(25, 100) == 0.1
    assert MODULE.learning_rate_factor(75, 100) == 0.05


def test_one_step_inversion_smoke():
    class Vision:
        def __call__(self, pixel_values, return_dict=True):
            pooled = pixel_values.mean(dim=(2, 3))
            return type("Output", (), {"pooler_output": pooled})()

    class FakeModel:
        def __init__(self):
            self.vision_model = Vision()
            self.visual_projection = torch.nn.Linear(3, 768, bias=False)
            self.visual_projection.requires_grad_(False)

    loader = [(torch.rand(2, 3, 336, 336), torch.zeros(2, dtype=torch.long))]
    result = MODULE.invert_trigger(
        FakeModel(), loader, device="cpu", epochs=1, lr=0.01, success_threshold=-1.0
    )
    assert result["label"] in (0, 1)
    assert result["mask"].shape == (1, 1, 336, 336)
    assert result["patch"].shape == (1, 3, 336, 336)
