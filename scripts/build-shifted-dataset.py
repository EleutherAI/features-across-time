import random
from collections import defaultdict
import io

import torch
from torch import Tensor
from torchvision.transforms import ToPILImage
from tqdm import tqdm
import numpy as np
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
import plotly.graph_objects as go
from scripts.script_utils.dury_distribution import DuryDistribution


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
        build_from_dataset(dataset)


def build_from_dataset(dataset: str, num_class_samples=1_000):
    path, _, name = dataset.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["test"].features)
    labels = ds["test"].features[label_col].names
    print(f"Classes in '{dataset}': {labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["test"][0][img_col]

    c, (h, w) = len(example.mode), example.size
    print(f"Image size: {h} x {w}")

    # Split out data by class
    class_data = defaultdict(list)
    for item in tqdm(ds["test"]):
        class_data[item[label_col]].append(item[img_col])

    # Calculate class mus
    # Path('tmp').mkdir(exist_ok=True)
    mus = []
    for label, data in class_data.items():
        scale = 1 / 255
        data = torch.stack([torch.tensor(np.array(img)) for img in data])
        mu = data.float().mean(dim=0) * scale
        mus.append(mu)
        # write_image(mu / scale, f"tmp/mu-{label}-{dataset}.png")

    classes = list(range(len(labels)))
    samples_list = []
    labels_list = []
    shifted_labels_list = []

    for current_class in classes:
        # Maximum entropy samples with means shifted to a random other class
        mu = mus[current_class]
        dd = DuryDistribution(mu, "cuda")
        samples = dd.sample(num_class_samples).cpu()

        shift_classes = random.choices([c for c in classes if c != current_class], k=num_class_samples)
        target_mus = torch.stack([mus[shift_class] for shift_class in shift_classes])
        shifted_samples = bounded_shift(samples, target_mus)
        shifted_samples /= scale
        
        img_bytes = io.BytesIO()
        for sample in shifted_samples:
            pil_img = ToPILImage()(sample.permute(2, 0, 1))
            pil_img.save(img_bytes, format='PNG')
            samples_list.append(img_bytes.getvalue())
        
        labels_list.extend([current_class] * num_class_samples)
        shifted_labels_list.extend(shift_classes)

    # match number of natural samples and max entropy samples? or just do 1000 max entropy
    data_dict = {
        'max_entropy_image': samples_list,
        'image': samples_list,
        'label': labels_list,
        'shifted_label': shifted_labels_list
    }

    dataset = Dataset.from_dict(data_dict)
    dataset.save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/{dataset}')
    # write_image(sample, f"tmp/sample-{label}-{dataset}.png")


if __name__ == "__main__":
    main()
    # Test over checkpoints in vision.py
