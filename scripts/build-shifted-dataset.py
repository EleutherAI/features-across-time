breakpoint()
import random
from random import choices
from collections import defaultdict
import os

import numpy as np
import torch
import torchvision.transforms.functional as TF
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from torch import Tensor
from tqdm import tqdm
import plotly.graph_objects as go
from script_utils.dury_distribution import DuryDistribution

def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def bounded_shift(
    x: Tensor, target: Tensor, bounds=(0., 1.), max_iter: int = 1000
):
    """Shift the elements of `x` s.t. their mean is `target` while keeping them within `bounds`."""

    # Initial centroid
    mu = x.mean(dim=0)

    for i in range(max_iter):
        x = x - (mu - target)
        x.clamp_(*bounds)

        mu = x.mean(dim=0)
        if torch.allclose(mu, target):
            print(f"Converged in {i} iterations")
            break
    
    return x


def write_image(tensor: Tensor, path: str):
    fig = go.Figure()
    fig.add_trace(
        go.Image(z=tensor)
    )
    fig.update_layout(
        xaxis=dict(visible=False),
        yaxis=dict(visible=False)
    )
    fig.write_image(path)


def main():
    datasets = ["cifar10", "svhn:cropped_digits", "mnist"]
    for dataset in datasets:
        build_from_dataset(dataset, num_class_samples=1_000, seed=0)


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

    if val := ds.get("validation"):
        test = ds["test"].with_transform(preprocess) if "test" in ds else None
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)
        test = nontrain["test"].with_transform(preprocess)

    # Split out data by class
    class_data = defaultdict(list)
    i = 0
    for item in tqdm(test):
        class_data[item["label"]].append(item["pixel_values"])
        i += 1
    print(i)
    
    # Calculate class mus
    mus = []
    for label in class_data.keys():
        class_data[label] = torch.stack([torch.tensor(np.array(img)) for img in class_data[label]])

        mu = class_data[label].float().mean(dim=0)
        mus.append(mu)

    classes = list(range(len(original_labels)))

    # Mean shifted natural samples
    shifted_samples = []
    original_labels = []
    shifted_labels = []
    for label, data in tqdm(class_data.items()):
        shift_classes = choices([c for c in classes if c != label], k=len(data))
        target_mus = torch.stack([mus[shift_class] for shift_class in shift_classes])
        
        print(target_mus.shape, data.shape)
        shifted_samples.append(bounded_shift(data, target_mus, bounds=(0, 1)))
        original_labels.extend([label] * len(data))
        shifted_labels.extend(shift_classes)
    
    data_dict = {
        'pixel_values': torch.cat(shifted_samples),
        'label': original_labels,
        # 'shifted_label': shifted_labels
    }
    Dataset.from_dict(data_dict).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/natural-{dataset}')

    # Mean shifted maximum entropy samples
    shifted_samples = []
    original_labels = []
    shifted_labels = []
    for label in tqdm(class_data.keys()):
        shift_classes = choices([c for c in classes if c != label], k=num_class_samples)
        target_mus = torch.stack([mus[shift_class] for shift_class in shift_classes])

        mu = mus[label]
        dd = DuryDistribution(mu, "cuda")
        samples = dd.sample(num_class_samples).cpu()
 
        shifted_sample_batch = bounded_shift(samples, target_mus, bounds=(0., 1.))
        shifted_samples.append(shifted_sample_batch)

        original_labels.extend([label] * num_class_samples)
        shifted_labels.extend(shift_classes)

        # Can convert to images if we need consistency with the original datasests
        # img_bytes = io.BytesIO()
        # for sample in shifted_samples:
        #     pil_img = ToPILImage()(sample.permute(2, 0, 1))
        #     pil_img.save(img_bytes, format='PNG')
        #     shifted_samples.append(img_bytes.getvalue())

    data_dict = {
        'pixel_values': torch.cat(shifted_samples),
        'label': original_labels,
        # 'shifted_label': shifted_labels
    }
    Dataset.from_dict(data_dict).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/max-entropy-{dataset}')


if __name__ == "__main__":
    # Path('tmp').mkdir(exist_ok=True)
    main()
    # Test over checkpoints in vision.py
