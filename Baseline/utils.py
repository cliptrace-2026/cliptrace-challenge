import torch
import torch.nn.functional as F


def compute_self_cos_sim(features):
    """Average cosine similarity between different samples in the same batch."""
    batch_size = features.shape[0]
    features = F.normalize(features, dim=-1)
    sim_matrix = torch.mm(features, features.t())

    off_diagonal_mask = ~torch.eye(batch_size, device=features.device, dtype=torch.bool)
    off_diagonal_values = sim_matrix.masked_select(off_diagonal_mask).view(batch_size, -1)
    return torch.mean(off_diagonal_values)


def assert_range(value, vmin, vmax, ratio=0.7):
    tensor = torch.as_tensor(value).float()
    actual_min = float(tensor.min().detach().cpu())
    actual_max = float(tensor.max().detach().cpu())

    vmin = vmin - 1e-4
    vmax = vmax + 1e-4
    diff = vmax - vmin
    if diff <= 0:
        raise ValueError(f"invalid range: [{vmin}, {vmax}]")

    min_ok = vmin <= actual_min <= vmin + ratio * diff
    max_ok = vmax - ratio * diff <= actual_max <= vmax
    if not (min_ok and max_ok):
        print("range warning:", f"value=[{actual_min}, {actual_max}], expected=[{vmin}, {vmax}], ratio={ratio}")


def epsilon():
    return 1e-7
