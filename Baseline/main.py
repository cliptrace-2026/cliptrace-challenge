import argparse
import json
import math
import os
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import DataLoader
from transformers import CLIPModel

try:
    from .imagenet import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD, get_clip_processing, get_tensor_imagenet
    from .utils import assert_range, compute_self_cos_sim, epsilon
except ImportError:
    from imagenet import CLIP_IMAGE_SIZE, CLIP_MEAN, CLIP_STD, get_clip_processing, get_tensor_imagenet
    from utils import assert_range, compute_self_cos_sim, epsilon


BASELINE_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = BASELINE_DIR.parent
PROJECTION_DIM = 768
PATCH_SIZE = 14
IMAGE_CHANNELS = 3
DEFAULT_PL1_THRESHOLD = 0.01


def project_path(path_like):
    path = Path(path_like).expanduser()
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def seed_everything(seed):
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def load_clip_model(encoder_path, device):
    model = CLIPModel.from_pretrained(str(encoder_path), local_files_only=True)
    vision_config = model.config.vision_config
    observed = (
        int(vision_config.image_size),
        int(vision_config.patch_size),
        int(model.config.projection_dim),
    )
    expected = (CLIP_IMAGE_SIZE[0], PATCH_SIZE, PROJECTION_DIM)
    if observed != expected:
        raise ValueError(f"unexpected CLIP config for {encoder_path.name}: {observed}, expected {expected}")

    model.eval()
    model.requires_grad_(False)
    return model.to(device)


def load_trigger(trigger_file):
    with np.load(trigger_file) as data:
        trigger_mask = np.asarray(data["tm"])
        trigger_patch = np.asarray(data["t"])

    if trigger_mask.ndim == 4:
        trigger_mask = trigger_mask[0]
    if trigger_patch.ndim == 4:
        trigger_patch = trigger_patch[0]

    if trigger_mask.shape != trigger_patch.shape:
        raise ValueError(f"trigger mask and patch shape mismatch: {trigger_mask.shape} != {trigger_patch.shape}")
    if trigger_mask.ndim != 3 or trigger_mask.shape[-1] != IMAGE_CHANNELS:
        raise ValueError(f"expected trigger shape (H, W, 3), got {trigger_mask.shape}")
    if trigger_mask.shape[:2] != CLIP_IMAGE_SIZE:
        raise ValueError(f"trigger size {trigger_mask.shape[:2]} does not match CLIP input size {CLIP_IMAGE_SIZE}")

    return trigger_mask, trigger_patch


def generate_square_mask(mask_size, top, left, size):
    mask = np.full((mask_size, mask_size), epsilon(), dtype=np.float64)
    bottom = min(top + size, mask_size)
    right = min(left + size, mask_size)
    mask[top:bottom, left:right] = 1.0
    return mask


def initialize_trigger(trigger_mask, trigger_patch, args, device):
    mask_size = trigger_mask.shape[0]
    if args.mask_init == "zero":
        train_mask_2d = torch.zeros(trigger_mask.shape[:2], dtype=torch.float64, device=device)
    elif args.mask_init == "orc":
        mask = generate_square_mask(mask_size, args.trigger_top, args.trigger_left, args.trigger_size)
        train_mask_2d = torch.tensor(mask, dtype=torch.float64, device=device)
        train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    elif args.mask_init == "rand":
        train_mask_2d = torch.rand(trigger_mask.shape[:2], dtype=torch.float64, device=device)
        train_mask_2d = torch.arctanh((train_mask_2d - 0.5) * (2 - epsilon()))
    else:
        raise ValueError(f"unsupported mask_init: {args.mask_init}")

    train_patch = torch.empty_like(torch.tensor(trigger_patch), dtype=torch.float64, device=device).uniform_(-1.0, 1.0)
    train_mask_2d.requires_grad = True
    train_patch.requires_grad = True
    return train_mask_2d, train_patch


def adjust_learning_rate(optimizer, epoch, args):
    progress = epoch / max(args.epochs, 1)
    if progress < 0.2:
        lr = args.lr
    elif progress < 0.5:
        lr = args.lr * 0.1
    else:
        lr = args.lr * 0.05

    print(f"epoch: {epoch}  lr: {lr:.4f}")
    for param_group in optimizer.param_groups:
        param_group["lr"] = lr


def projected_features(model, normalized_images):
    features = model.get_image_features(pixel_values=normalized_images)
    return F.normalize(features, dim=-1)


def prepare_batch(clean_x_batch, mask_2d, patch, transform, device):
    mask_3d = mask_2d.unsqueeze(2).repeat(1, 1, 3)
    mask = torch.tanh(mask_3d) / (2 - epsilon()) + 0.5
    patch = (torch.tanh(patch) / (2 - epsilon()) + 0.5) * 255
    mask = torch.clip(mask, min=0, max=1)
    patch = torch.clip(patch, min=0, max=255)

    poisoned_x_batch = (1 - mask) * clean_x_batch + mask * patch
    poisoned_x_batch = torch.clip(poisoned_x_batch, min=0, max=255)

    poisoned_input = []
    for image in poisoned_x_batch:
        poisoned_input.append(transform(image.permute(2, 0, 1) / 255.0))
    poisoned_input = torch.stack(poisoned_input).to(dtype=torch.float, device=device)

    return poisoned_input, mask, patch


def save_trigger_artifacts(output_dir, suffix, mask, patch):
    output_dir.mkdir(parents=True, exist_ok=True)
    fusion = np.asarray(mask * patch, dtype=np.uint8)
    mask_img = np.asarray(mask * 255, dtype=np.uint8)
    patch_img = np.asarray(patch, dtype=np.uint8)

    Image.fromarray(mask_img).save(output_dir / f"mask_{suffix}.png")
    Image.fromarray(patch_img).save(output_dir / f"patch_{suffix}.png")
    Image.fromarray(fusion).save(output_dir / f"fus_{suffix}.png")


def recover_submission_embedding(model, loader, mask_np, patch_np, transform, device):
    mask = torch.tensor(mask_np, dtype=torch.float32, device=device)
    patch = torch.tensor(patch_np, dtype=torch.float32, device=device)
    features = []

    with torch.no_grad():
        for clean_x_batch, _ in loader:
            clean_x_batch = clean_x_batch.to(device)
            poisoned_x_batch = (1 - mask) * clean_x_batch + mask * patch
            poisoned_x_batch = torch.clip(poisoned_x_batch, min=0, max=255)

            poisoned_input = []
            for image in poisoned_x_batch:
                poisoned_input.append(transform(image.permute(2, 0, 1) / 255.0))
            poisoned_input = torch.stack(poisoned_input).to(dtype=torch.float, device=device)
            features.append(projected_features(model, poisoned_input).cpu())

    embedding = F.normalize(torch.cat(features).mean(dim=0), dim=0)
    embedding = embedding.to(device="cpu", dtype=torch.float32).reshape(PROJECTION_DIM)
    if not torch.isfinite(embedding).all() or not math.isclose(float(embedding.norm()), 1.0, abs_tol=1e-4):
        raise RuntimeError("submission embedding is not finite and L2-normalized")
    return embedding


def read_predictions(predictions_path):
    if not predictions_path.is_file():
        return {}
    payload = json.loads(predictions_path.read_text(encoding="utf-8"))
    return {item["model_id"]: item for item in payload.get("predictions", [])}


def write_predictions(predictions_path, predictions):
    payload = {"version": "1.0", "predictions": [predictions[key] for key in sorted(predictions)]}
    predictions_path.parent.mkdir(parents=True, exist_ok=True)
    predictions_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")


def write_submission_prediction(submission_dir, model_id, label, embedding):
    embeddings_dir = submission_dir / "embeddings"
    embeddings_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = submission_dir / "predictions.json"
    predictions = read_predictions(predictions_path)

    embedding_file = None
    embedding_path = embeddings_dir / f"{model_id}.pt"
    if label == 1:
        if embedding is None:
            raise ValueError(f"model {model_id} is labeled backdoor but has no embedding")
        torch.save(embedding.to(device="cpu", dtype=torch.float32), embedding_path)
        embedding_file = f"embeddings/{embedding_path.name}"
    elif embedding_path.exists():
        embedding_path.unlink()

    predictions[model_id] = {
        "model_id": model_id,
        "label": int(label),
        "embedding_file": embedding_file,
    }
    write_predictions(predictions_path, predictions)


def append_metrics(metrics_file, metrics):
    metrics_file.parent.mkdir(parents=True, exist_ok=True)
    with metrics_file.open("a", encoding="utf-8") as fp:
        fp.write(json.dumps(metrics, ensure_ascii=False, allow_nan=False) + "\n")


def main(args):
    if args.batch_size < 2:
        raise ValueError("batch_size must be at least 2 because self-cosine similarity compares samples in a batch")

    seed_everything(args.seed)

    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    if device.type == "cpu":
        print("Warning: CUDA is not available, running on CPU will be slow.")

    encoder_path = project_path(args.encoder_path)
    imagenet_root = project_path(args.imagenet_root)
    trigger_file = project_path(args.trigger_file)
    result_dir = project_path(args.result_dir)
    submission_dir = project_path(args.submission_dir)
    metrics_file = project_path(args.metrics_file)
    model_id = args.model_id or encoder_path.name

    model = load_clip_model(encoder_path, device)
    trigger_mask, trigger_patch = load_trigger(trigger_file)
    pre_transform = get_clip_processing(augment=False, need_norm=False)
    test_transform = transforms.Compose([
        transforms.Resize(CLIP_IMAGE_SIZE, antialias=True),
        transforms.Normalize(CLIP_MEAN, CLIP_STD),
    ])

    clean_train_data = get_tensor_imagenet(imagenet_root, pre_transform, split=args.data_split)
    clean_train_data.rand_sample(args.sample_ratio)
    clean_train_loader = DataLoader(
        clean_train_data,
        batch_size=args.batch_size,
        pin_memory=(device.type == "cuda"),
        shuffle=True,
        num_workers=args.num_workers,
    )

    train_mask_2d, train_patch = initialize_trigger(trigger_mask, trigger_patch, args, device)
    optimizer = torch.optim.Adam(params=[train_mask_2d, train_patch], lr=args.lr, betas=(0.5, 0.9))

    init_loss_lambda = 1e-3
    loss_lambda = init_loss_lambda
    adaptor_lambda = 5.0
    patience = 5
    adaptor_up_cnt = 0
    adaptor_down_cnt = 0
    lambda_set_cnt = 0
    lambda_set_patience = 2 * patience
    succ_threshold = args.thres
    regular_best = float("inf")
    early_stop_reg_best = regular_best
    early_stop_cnt = 0
    best_trigger = None
    best_cos = float("-inf")

    if args.encoder_usage_info in ["HF_CLIP", "imagenet", "moco"]:
        lambda_min = 1e-7
        early_stop_patience = 7 * patience
    elif args.encoder_usage_info in ["cifar10", "stl10"]:
        lambda_min = 1e-5
        early_stop_patience = 2 * patience
    else:
        raise ValueError(f"unsupported encoder_usage_info: {args.encoder_usage_info}")

    print(f"model_id: {model_id}")
    print(f"encoder: {encoder_path}")
    print(f"trigger: {trigger_file}")
    print(f"imagenet: {imagenet_root}")
    print(f"submission: {submission_dir}")
    print(f"dataset size: {len(clean_train_data)}")
    print(
        f"Config: lambda_min={lambda_min}, adapt_lambda={adaptor_lambda}, "
        f"lambda_set_patience={lambda_set_patience}, succ_threshold={succ_threshold}, "
        f"pl1_threshold={args.pl1_threshold}, early_stop_patience={early_stop_patience}"
    )

    regular_list, cosine_list = [], []
    start_time = time.time()
    artifact_dir = result_dir / model_id
    stopped_early = False
    stop_reason = None

    for epoch in range(args.epochs):
        adjust_learning_rate(optimizer, epoch, args)
        loss_values, cos_values, reg_values = [], [], []

        for step, (clean_x_batch, _) in enumerate(clean_train_loader):
            if clean_x_batch.shape[0] < 2:
                continue
            assert clean_x_batch.shape[-1] == 3
            clean_x_batch = clean_x_batch.to(device)
            assert_range(clean_x_batch, 0, 255)

            poisoned_input, mask, patch = prepare_batch(clean_x_batch, train_mask_2d, train_patch, test_transform, device)
            assert_range(poisoned_input, -3, 3)

            poisoned_features = projected_features(model, poisoned_input)
            cos_sim = compute_self_cos_sim(poisoned_features)
            loss_cos = -cos_sim
            loss_reg = torch.sum(torch.abs(mask))
            loss = loss_cos + loss_reg * loss_lambda

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            loss_values.append(loss.detach())
            cos_values.append(cos_sim.detach())
            reg_values.append(loss_reg.detach())

            cos_value = float(cos_sim.detach().cpu())
            reg_value = float(loss_reg.detach().cpu())
            best_cos = max(best_cos, cos_value)
            if cos_value >= succ_threshold and reg_value < regular_best:
                best_trigger = {
                    "mask": mask.detach().cpu().numpy(),
                    "patch": patch.detach().cpu().numpy(),
                    "similarity": cos_value,
                }
                regular_best = reg_value

                current_pl1 = regular_best / float(CLIP_IMAGE_SIZE[0] * CLIP_IMAGE_SIZE[1] * IMAGE_CHANNELS)
                if args.stop_on_backdoor_pl1 and current_pl1 < args.pl1_threshold:
                    stop_reason = "pl1_threshold_reached"
                    print(f"Stop: PL1={current_pl1:.6f} < threshold={args.pl1_threshold:.6f}")
                    break

            if regular_best < float("inf"):
                if regular_best >= early_stop_reg_best:
                    early_stop_cnt += 1
                else:
                    early_stop_cnt = 0
            early_stop_reg_best = min(regular_best, early_stop_reg_best)

            if loss_lambda < lambda_min and cos_value >= succ_threshold:
                lambda_set_cnt += 1
                if lambda_set_cnt > lambda_set_patience:
                    loss_lambda = init_loss_lambda
                    adaptor_up_cnt = 0
                    adaptor_down_cnt = 0
                    print(f"step {step}: reset loss_lambda to {loss_lambda}")
            else:
                lambda_set_cnt = 0

            if cos_value >= succ_threshold:
                adaptor_up_cnt += 1
                adaptor_down_cnt = 0
            else:
                adaptor_down_cnt += 1
                adaptor_up_cnt = 0

            if adaptor_up_cnt > patience:
                if loss_lambda < 1e5:
                    loss_lambda *= adaptor_lambda
                adaptor_up_cnt = 0
                print(f"step {step}: loss_lambda is up to {loss_lambda}")
            elif adaptor_down_cnt > patience:
                if loss_lambda >= lambda_min:
                    loss_lambda /= adaptor_lambda
                adaptor_down_cnt = 0
                print(f"step {step}: loss_lambda is down to {loss_lambda}")

        if stop_reason is not None:
            stopped_early = True

        if not loss_values:
            raise ValueError("no valid batch was processed; lower batch_size or increase sample_ratio")

        loss_avg = torch.mean(torch.stack(loss_values))
        cos_avg = torch.mean(torch.stack(cos_values))
        reg_avg = torch.mean(torch.stack(reg_values))
        success_reg_best = "NA" if not math.isfinite(regular_best) else f"{regular_best:.6f}"
        early_stop_best = "NA" if not math.isfinite(early_stop_reg_best) else f"{early_stop_reg_best:.6f}"
        print(
            f"e={epoch}, loss={loss_avg:.6f}, cos={cos_avg:.6f}, "
            f"loss_reg={reg_avg:.6f}, success_reg_best={success_reg_best}, "
            f"best_cos={best_cos:.6f}, es_reg_best={early_stop_best}"
        )
        regular_list.append(str(round(float(reg_avg), 2)))
        cosine_list.append(str(round(float(cos_avg), 2)))

        if best_trigger is not None:
            assert_range(best_trigger["mask"], 0, 1)
            assert_range(best_trigger["patch"], 0, 255)
            suffix = f"e{epoch}_reg{regular_best:.2f}"
            save_trigger_artifacts(artifact_dir, suffix, best_trigger["mask"], best_trigger["patch"])

        if stop_reason is not None:
            break

        if float(cos_avg) >= succ_threshold and early_stop_cnt > early_stop_patience:
            print("Early stop!")
            stopped_early = True
            stop_reason = "patience"
            break

    duration = time.time() - start_time
    pl1 = regular_best / float(CLIP_IMAGE_SIZE[0] * CLIP_IMAGE_SIZE[1] * IMAGE_CHANNELS)
    if not math.isfinite(pl1):
        pl1 = 1.0

    label = int(best_trigger is not None and pl1 < args.pl1_threshold)
    embedding = None
    if label == 1:
        embedding = recover_submission_embedding(
            model,
            clean_train_loader,
            best_trigger["mask"],
            best_trigger["patch"],
            test_transform,
            device,
        )

    write_submission_prediction(submission_dir, model_id, label, embedding)
    metrics = {
        "model_id": model_id,
        "encoder_path": str(encoder_path),
        "label": label,
        "pl1": pl1,
        "l1": regular_best if math.isfinite(regular_best) else None,
        "best_cos": best_cos if math.isfinite(best_cos) else None,
        "similarity": None if best_trigger is None else best_trigger["similarity"],
        "duration_seconds": duration,
        "stopped_early": stopped_early,
        "stop_reason": stop_reason,
    }
    append_metrics(metrics_file, metrics)

    print(f"End:{duration:.4f}s")
    print(f"label:{label} pl1:{pl1:.6f}")
    print("reg:", ",".join(regular_list))
    print("cos:", ",".join(cosine_list))
    return metrics


def parse_args():
    parser = argparse.ArgumentParser(description="Reverse a possible CLIP backdoor trigger and write CLIPTrace submission files.")
    parser.add_argument("--batch_size", default=12, type=int, help="number of images in each mini-batch")
    parser.add_argument("--gpu", default="0", type=str, help="GPU id")
    parser.add_argument("--lr", default=0.1, type=float, help="learning rate for trigger optimization")
    parser.add_argument("--seed", default=42, type=int, help="random seed")
    parser.add_argument("--model_id", type=str, help="model id written to predictions.json; defaults to encoder directory name")
    parser.add_argument(
        "--encoder_path",
        default="cliptrace-2026-models/models/development/model_0001",
        type=str,
        help="relative or absolute path to the CLIP model",
    )
    parser.add_argument("--encoder_usage_info", default="HF_CLIP", type=str, help="HF_CLIP/imagenet/moco/cifar10/stl10")
    parser.add_argument("--mask_init", default="zero", choices=["zero", "rand", "orc"], help="trigger mask initialization")
    parser.add_argument("--trigger_top", default=24, type=int, help="top position used by --mask_init orc")
    parser.add_argument("--trigger_left", default=24, type=int, help="left position used by --mask_init orc")
    parser.add_argument("--trigger_size", default=176, type=int, help="square size used by --mask_init orc")
    parser.add_argument("--imagenet_root", default="Baseline/data/imagenet", type=str, help="ImageNet root or its parent")
    parser.add_argument("--trigger_file", default="Baseline/trigger/trigger_clip_l.npz", type=str, help="trigger npz file")
    parser.add_argument("--result_dir", default="outputs/baseline-artifacts", type=str, help="directory for debug trigger images")
    parser.add_argument("--submission_dir", default="submission", type=str, help="directory containing predictions.json and embeddings/")
    parser.add_argument("--metrics_file", default="outputs/baseline_metrics.jsonl", type=str, help="jsonl metrics log")
    parser.add_argument("--data_split", default="val", choices=["train", "val"], help="ImageNet split")
    parser.add_argument("--sample_ratio", default=0.2, type=float, help="fraction of ImageNet split to use")
    parser.add_argument("--epochs", default=100, type=int, help="maximum optimization epochs")
    parser.add_argument("--num_workers", default=0, type=int, help="DataLoader worker count")
    parser.add_argument("--thres", default=0.99, type=float, help="success threshold for feature cosine similarity")
    parser.add_argument("--pl1_threshold", default=DEFAULT_PL1_THRESHOLD, type=float, help="label=1 if PL1 is below this value")
    parser.add_argument(
        "--stop_on_backdoor_pl1",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="stop optimizing a model as soon as a successful trigger also satisfies the PL1 backdoor threshold",
    )
    return parser.parse_args()


if __name__ == "__main__":
    parsed_args = parse_args()
    print(parsed_args)
    main(parsed_args)
