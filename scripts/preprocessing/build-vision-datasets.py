import random
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from torch import Tensor
from tqdm import tqdm
from script_utils.dury_distribution import DuryDistribution
from concept_erasure.utils import assert_type
from einops import rearrange


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def bounded_shift(
    x: Tensor, target: Tensor, bounds=(0., 1.), max_iter: int = 10_000
):
    """Shift the elements of `x` s.t. their mean is `target` while keeping them within `bounds`."""
    # Initial centroid
    mu = x.mean(dim=0)

    for i in tqdm(range(max_iter)):
        x = x - (mu - target)
        x.clamp_(*bounds)

        mu = x.mean(dim=0)
        if torch.allclose(mu, target):
            print(f"Converged in {i} iterations")
            break

    return x


def build_from_dataset(dataset_str: str, seed: int):
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    # Allow specifying load_dataset("svhn", "cropped_digits") as "svhn:cropped_digits"
    # We don't use the slash because it's a valid character in a dataset name
    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    def preprocess(batch):
        return {
            "pixel_values": [TF.to_tensor(x) for x in batch[img_col]],
            "label": torch.tensor(batch[label_col]),
        }

    if val := ds.get("validation"):
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)

    with val.formatted_as("torch"):
        X = assert_type(Tensor, val[img_col]).div(255)
        X = rearrange(X, "n h w c -> n c h w")
        Y = assert_type(Tensor, val[label_col])

    # from torchvision.transforms.functional import to_pil_image
    # for i in range(3):
    #     image = to_pil_image(X[i])
    #     image.save(f'images/tmp-svhn-{dataset_str}-{i}.png')
    # breakpoint()

    class_data = defaultdict(list)
    for i in range(len(X)):
        class_data[Y[i].item()].append(X[i])

    mus = {
        key: torch.stack(data).float().mean(dim=0) for key, data in class_data.items()
    }

    print("Generating mean shifted data...")
    shifted_data = []
    labels = []
    original_labels = []
    for label, data in tqdm(class_data.items()):
        targets = {i: mu for i, mu in mus.items() if i != label}
        
        for i, mu in targets.items():
            shifted = bounded_shift(torch.stack(data), mu, bounds=(0., 1.))
            labels.extend([i] * len(data))
            original_labels.extend([label] * len(data))
            shifted_data.append(shifted)
    
    data_dict = {
        'pixel_values': torch.cat(shifted_data),
        'original': torch.tensor(original_labels),
        'label': torch.tensor(labels)
    }
    Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/shifted-{dataset_str}.hf')

    print("Generating maximum entropy data...")
    dd_data = []
    labels = []
    for label, original_data in tqdm(class_data.items()):
        mu = mus[label].numpy()
        dd = DuryDistribution(mu)
        data = dd.sample(len(original_data))
        dd_data.append(torch.tensor(data, dtype=torch.float))
        labels.extend([label] * len(original_data))
    
    data_dict = {
        'pixel_values': torch.cat(dd_data),
        'label': torch.tensor(labels)
    }
    Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/max-entropy-{dataset_str}.hf')


def main():
    datasets = [
        # "cifar10", 
        "svhn:cropped_digits", 
        # "mnist",
        # "evanarlian/imagenet_1k_resized_256", 
        # "fashion_mnist",
        # "cifarnet",
    ]
    for dataset in datasets:
        build_from_dataset(dataset, seed=0)


if __name__ == "__main__":
    main()
