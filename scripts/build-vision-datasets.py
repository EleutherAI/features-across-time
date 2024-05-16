import random
from collections import defaultdict

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from torch import Tensor
from tqdm import tqdm
from script_utils.dury_distribution import DuryDistribution


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


def build_from_dataset(dataset: str, num_class_samples: int, seed: int):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    path, _, name = dataset.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["test"].features)
    original_labels = ds["test"].features[label_col].names
    print(f"Classes in '{dataset}': {original_labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["test"][0][img_col]

    c, (h, w) = len(example.mode), example.size
    print(f"Image size: {h} x {w}")

    def preprocess(batch):
        return {
            "pixel_values": [TF.to_tensor(x) for x in batch[img_col]],
            "label": torch.tensor(batch[label_col]),
        }

    nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
    test = nontrain["test"].with_transform(preprocess)
    class_data = defaultdict(list)
    for item in tqdm(test):
        class_data[item["label"].item()].append(item["pixel_values"])
    del test

    mus = {
        key: torch.stack(data).float().mean(dim=0) for key, data in class_data.items()
    }

    print("Generating mean shifted data...")
    shifted_data = []
    shifted_labels = []
    original_labels = []
    for label, data in tqdm(class_data.items()):
        targets = {i: mu for i, mu in mus.items() if i != label}
        original_labels.extend([label] * (len(data) - len(data) % len(list(targets.keys()))))
        
        for idx, (i, mu) in enumerate(targets.items()):
            batch = data[
                idx * (len(data) // len(targets)):
                idx * (len(data) // len(targets)) + len(data) // len(targets)
            ]
            shifted = bounded_shift(torch.stack(batch), mu, bounds=(0., 1.))
            shifted_labels.extend([i] * len(batch))
            shifted_data.append(shifted)
    
    data_dict = {
        'pixel_values': torch.cat(shifted_data),
        'label': torch.tensor(original_labels),
        'target': torch.tensor(shifted_labels)
    }
    Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/natural-{dataset}.hf')

    print("Generating maximum entropy data...")
    dd_data = []
    original_labels = []
    for label in tqdm(class_data.keys()):
        mu = mus[label].numpy()
        dd = DuryDistribution(mu, start=0.1)
        data = dd.sample(num_class_samples)
        # resample until there's no nans
        mask = ~np.isfinite(data)
        i = 0
        while mask.sum() > 0:
            i += 1
            dd = DuryDistribution(mu, start=random.uniform(0, 1))
            data[mask] = dd.sample(num_class_samples)[mask]
            mask = ~np.isfinite(data)
        print("i", i)
        dd_data.append(torch.tensor(data, dtype=torch.float))
        original_labels.extend([label] * num_class_samples)
    
    # torch.save(dd_data, f'/mnt/ssd-1/lucia/shifted-data/max-entropy-{dataset}.pt')
    data_dict = {
        'pixel_values': torch.cat(dd_data),
        'label': torch.tensor(original_labels)
    }
    Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/max-entropy-{dataset}.hf')


def main():
    datasets = [
        # "cifar10", 
        # "svhn:cropped_digits", 
        # "mnist"
        "evanarlian/imagenet_1k_resized_256", 
        "fashion_mnist",
        "cifarnet"
    ]
    for dataset in datasets:
        build_from_dataset(dataset, num_class_samples=1_000, seed=0)


if __name__ == "__main__":
    main()
