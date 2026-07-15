#!/usr/bin/env python3
"""DECREE-style CLIP backdoor detection and target-feature recovery baseline."""

from __future__ import annotations

import argparse
import json
import math
import os
import random
import time
from pathlib import Path


IMAGE_SIZE = 336
PATCH_SIZE = 14
PROJECTION_DIM = 768
PL1_THRESHOLD = 0.10
CLIP_MEAN = (0.48145466, 0.4578275, 0.40821073)
CLIP_STD = (0.26862954, 0.26130258, 0.27577711)


def seed_everything(seed: int) -> None:
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def resolve_device(requested: str) -> str:
    import torch

    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested, but no CUDA device is available")
    return requested


def find_validation_split(data_dir: Path) -> Path:
    for candidate in (data_dir / "val", data_dir / "imagenet" / "val"):
        if candidate.is_dir():
            return candidate
    raise FileNotFoundError(f"Could not find val/ under {data_dir}")


def build_loader(data_dir: Path, *, batch_size: int, max_samples: int, seed: int, device: str):
    from torch.utils.data import DataLoader, Subset
    from torchvision import datasets, transforms

    transform = transforms.Compose(
        [
            transforms.Resize(IMAGE_SIZE, antialias=True),
            transforms.CenterCrop(IMAGE_SIZE),
            transforms.ToTensor(),
        ]
    )
    dataset = datasets.ImageFolder(find_validation_split(data_dir), transform=transform)
    indices = list(range(len(dataset)))
    random.Random(seed).shuffle(indices)
    indices = sorted(indices[: min(max_samples, len(indices))])
    if len(indices) < 2:
        raise ValueError("The baseline needs at least two images")
    subset = Subset(dataset, indices)
    return DataLoader(
        subset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=device.startswith("cuda"),
    )


def load_model(model_dir: Path, device: str):
    from transformers import CLIPModel

    model = CLIPModel.from_pretrained(str(model_dir), local_files_only=True)
    vision = model.config.vision_config
    observed = (int(vision.image_size), int(vision.patch_size), int(model.config.projection_dim))
    expected = (IMAGE_SIZE, PATCH_SIZE, PROJECTION_DIM)
    if observed != expected:
        raise ValueError(f"Unexpected CLIP configuration for {model_dir.name}: {observed}, expected {expected}")
    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def normalize_pixels(images):
    import torch

    mean = torch.tensor(CLIP_MEAN, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    std = torch.tensor(CLIP_STD, device=images.device, dtype=images.dtype).view(1, 3, 1, 1)
    return (images - mean) / std


def projected_features(model, images):
    import torch.nn.functional as functional

    outputs = model.vision_model(pixel_values=normalize_pixels(images), return_dict=True)
    features = model.visual_projection(outputs.pooler_output)
    return functional.normalize(features, dim=-1)


def off_diagonal_similarity(features):
    import torch

    count = features.shape[0]
    if count < 2:
        raise ValueError("A batch must contain at least two images")
    similarities = features @ features.T
    keep = ~torch.eye(count, dtype=torch.bool, device=features.device)
    return similarities.masked_select(keep).mean()


def learning_rate_factor(epoch: int, epochs: int) -> float:
    progress = epoch / max(epochs, 1)
    if progress < 0.2:
        return 1.0
    if progress < 0.5:
        return 0.1
    return 0.05


def recover_target_feature(model, loader, mask, patch, device: str):
    import torch
    import torch.nn.functional as functional

    mask = mask.to(device=device, dtype=torch.float32)
    patch = patch.to(device=device, dtype=torch.float32)
    features = []
    with torch.no_grad():
        for images, _ in loader:
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            triggered = (1.0 - mask) * images + mask * patch
            features.append(projected_features(model, triggered).cpu())
    target = functional.normalize(torch.cat(features).mean(dim=0), dim=0)
    target = target.to(device="cpu", dtype=torch.float32).reshape(PROJECTION_DIM)
    if not torch.isfinite(target).all() or not math.isclose(float(target.norm()), 1.0, abs_tol=1e-4):
        raise RuntimeError("Recovered target feature is not finite and L2-normalized")
    return target


def invert_trigger(model, loader, *, device: str, epochs: int, lr: float, success_threshold: float):
    import torch

    mask_logits = torch.zeros((1, 1, IMAGE_SIZE, IMAGE_SIZE), device=device, requires_grad=True)
    patch_logits = torch.empty((1, 3, IMAGE_SIZE, IMAGE_SIZE), device=device).uniform_(-1.0, 1.0)
    patch_logits.requires_grad_(True)
    optimizer = torch.optim.Adam([mask_logits, patch_logits], lr=lr, betas=(0.5, 0.9))

    loss_lambda = 1e-3
    lambda_min = 1e-7
    patience = 5
    increase_count = decrease_count = 0
    best = None

    for epoch in range(epochs):
        for group in optimizer.param_groups:
            group["lr"] = lr * learning_rate_factor(epoch, epochs)

        epoch_similarities = []
        for images, _ in loader:
            if images.shape[0] < 2:
                continue
            images = images.to(device=device, dtype=torch.float32, non_blocking=True)
            mask = torch.sigmoid(mask_logits)
            patch = torch.sigmoid(patch_logits)
            triggered = (1.0 - mask) * images + mask * patch

            similarity = off_diagonal_similarity(projected_features(model, triggered))
            l1 = mask.sum() * 3.0
            loss = -similarity + loss_lambda * l1

            optimizer.zero_grad(set_to_none=True)
            loss.backward()
            optimizer.step()

            similarity_value = float(similarity.detach().cpu())
            l1_value = float(l1.detach().cpu())
            epoch_similarities.append(similarity_value)
            if similarity_value >= success_threshold and (best is None or l1_value < best[0]):
                best = (
                    l1_value,
                    similarity_value,
                    mask.detach().cpu().clone(),
                    patch.detach().cpu().clone(),
                )

            if similarity_value >= success_threshold:
                increase_count += 1
                decrease_count = 0
                if increase_count > patience:
                    loss_lambda = min(loss_lambda * 5.0, 1e5)
                    increase_count = 0
            else:
                decrease_count += 1
                increase_count = 0
                if decrease_count > patience:
                    loss_lambda = max(loss_lambda / 5.0, lambda_min)
                    decrease_count = 0

        mean_similarity = sum(epoch_similarities) / max(len(epoch_similarities), 1)
        print(f"  epoch {epoch + 1:03d}/{epochs}: similarity={mean_similarity:.4f}, lambda={loss_lambda:.2e}")

    if best is None:
        return {
            "label": 0,
            "pl1": 1.0,
            "similarity": None,
            "mask": torch.sigmoid(mask_logits).detach().cpu(),
            "patch": torch.sigmoid(patch_logits).detach().cpu(),
            "embedding": None,
        }

    l1, similarity, mask, patch = best
    pl1 = l1 / float(IMAGE_SIZE * IMAGE_SIZE * 3)
    label = int(pl1 < PL1_THRESHOLD)
    embedding = recover_target_feature(model, loader, mask, patch, device) if label else None
    return {
        "label": label,
        "pl1": pl1,
        "similarity": similarity,
        "mask": mask,
        "patch": patch,
        "embedding": embedding,
    }


def discover_models(models_dir: Path, model_id: str | None) -> list[Path]:
    if model_id:
        path = models_dir / model_id
        if not (path / "config.json").is_file():
            raise FileNotFoundError(f"Model checkpoint not found: {path}")
        return [path]
    models = sorted(path for path in models_dir.iterdir() if path.is_dir() and (path / "config.json").is_file())
    if not models:
        raise FileNotFoundError(f"No Hugging Face model directories found under {models_dir}")
    return models


def read_existing_predictions(path: Path) -> dict[str, dict]:
    if not path.is_file():
        return {}
    payload = json.loads(path.read_text(encoding="utf-8"))
    return {item["model_id"]: item for item in payload.get("predictions", [])}


def write_predictions(path: Path, predictions: dict[str, dict]) -> None:
    payload = {"version": "1.0", "predictions": [predictions[key] for key in sorted(predictions)]}
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def main() -> int:
    parser = argparse.ArgumentParser(description="Run the CLIPTrace DECREE-style baseline")
    parser.add_argument("--models-dir", type=Path, required=True)
    parser.add_argument("--data-dir", type=Path, required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    parser.add_argument("--model-id", help="process only one model and update the existing predictions file")
    parser.add_argument("--device", default="auto")
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--max-samples", type=int, default=785)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--lr", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--success-threshold", type=float, default=0.99)
    args = parser.parse_args()
    if args.epochs < 1 or args.max_samples < 2 or args.batch_size < 2:
        parser.error("epochs must be >= 1; max-samples and batch-size must be >= 2")

    seed_everything(args.seed)
    device = resolve_device(args.device)
    models = discover_models(args.models_dir, args.model_id)
    loader = build_loader(
        args.data_dir,
        batch_size=args.batch_size,
        max_samples=args.max_samples,
        seed=args.seed,
        device=device,
    )

    submission_dir = args.submission_dir.resolve()
    embeddings_dir = submission_dir / "embeddings"
    artifacts_dir = submission_dir.parent / "outputs" / "baseline-artifacts"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    artifacts_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = submission_dir / "predictions.json"
    predictions = read_existing_predictions(predictions_path) if args.model_id else {}
    metrics = []

    for index, model_path in enumerate(models, start=1):
        model_id = model_path.name
        print(f"\n[{index}/{len(models)}] Processing {model_id} on {device}")
        started = time.perf_counter()
        model = load_model(model_path, device)
        result = invert_trigger(
            model,
            loader,
            device=device,
            epochs=args.epochs,
            lr=args.lr,
            success_threshold=args.success_threshold,
        )

        model_artifacts = artifacts_dir / model_id
        model_artifacts.mkdir(parents=True, exist_ok=True)
        import torch

        torch.save(result["mask"].to(dtype=torch.float32, device="cpu"), model_artifacts / "mask.pt")
        torch.save(result["patch"].to(dtype=torch.float32, device="cpu"), model_artifacts / "patch.pt")

        embedding_file = None
        if result["label"] == 1:
            target = embeddings_dir / f"{model_id}.pt"
            torch.save(result["embedding"], target)
            embedding_file = f"embeddings/{target.name}"
        else:
            stale = embeddings_dir / f"{model_id}.pt"
            if stale.exists():
                stale.unlink()

        predictions[model_id] = {
            "model_id": model_id,
            "label": result["label"],
            "embedding_file": embedding_file,
        }
        write_predictions(predictions_path, predictions)
        duration = time.perf_counter() - started
        metrics.append(
            {
                "model_id": model_id,
                "label": result["label"],
                "pl1": result["pl1"],
                "similarity": result["similarity"],
                "duration_seconds": duration,
            }
        )
        print(f"  label={result['label']} pl1={result['pl1']:.6f} duration={duration:.1f}s")
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    log_path = artifacts_dir.parent / "baseline_metrics.json"
    log_path.write_text(json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8")
    print(f"\nPredictions written to {predictions_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

