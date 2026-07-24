import random
from pathlib import Path

import torch
import torchvision
import torchvision.transforms as transforms
from PIL import Image
from torch.utils.data import Dataset


CLIP_MEAN = [0.48145466, 0.4578275, 0.40821073]
CLIP_STD = [0.26862954, 0.26130258, 0.27577711]
CLIP_IMAGE_SIZE = (336, 336)


class ImageNetTensorDataset(Dataset):
    """ImageFolder wrapper that returns unnormalized uint8 tensors in HWC layout."""

    def __init__(self, dataset, transform):
        if not isinstance(dataset, Dataset):
            raise TypeError("dataset must be a torch Dataset")
        if transform is None:
            raise ValueError("transform must not be None")

        self.targets = list(dataset.targets)
        self.filenames = [sample[0] for sample in dataset.samples]
        self.transform = transform

    def __getitem__(self, index):
        image = Image.open(self.filenames[index]).convert("RGB")
        image = self.transform(image)
        image = image.to(dtype=torch.float64)
        image = (image.permute(1, 2, 0) * 255).type(torch.uint8)
        return image, self.targets[index]

    def __len__(self):
        return len(self.targets)

    def rand_sample(self, ratio):
        if not 0 < ratio <= 1:
            raise ValueError(f"sample ratio must be in (0, 1], got {ratio}")
        if ratio == 1:
            return

        sample_size = max(2, int(len(self.targets) * ratio))
        sample_size = min(sample_size, len(self.targets))
        indices = random.sample(range(len(self.targets)), sample_size)
        self.targets = [self.targets[index] for index in indices]
        self.filenames = [self.filenames[index] for index in indices]


def find_split_root(root, split):
    root = Path(root)
    candidates = [
        root / split,
        root / "imagenet" / split,
        root,
    ]
    for candidate in candidates:
        if candidate.is_dir() and any(path.is_dir() for path in candidate.iterdir()):
            return candidate
    raise FileNotFoundError(f"could not find ImageFolder split '{split}' under {root}")


def get_tensor_imagenet(root, transform, split="val"):
    if split not in ["train", "val"]:
        raise ValueError(f"split must be 'train' or 'val', got {split}")

    dataset = torchvision.datasets.ImageFolder(find_split_root(root, split), transform=None)
    return ImageNetTensorDataset(dataset, transform)


def get_clip_processing(augment=False, need_norm=True):
    transform_list = []
    if augment:
        transform_list.append(transforms.RandomResizedCrop(CLIP_IMAGE_SIZE, scale=(0.2, 1.0)))
        transform_list.append(transforms.RandomHorizontalFlip())
    else:
        transform_list.append(transforms.Resize(384))
        transform_list.append(transforms.CenterCrop(CLIP_IMAGE_SIZE))

    transform_list.append(transforms.ToTensor())
    if need_norm:
        transform_list.append(transforms.Normalize(CLIP_MEAN, CLIP_STD))

    return transforms.Compose(transform_list)
