import random
from random import choices
from collections import defaultdict

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
    x: Tensor, target: Tensor, bounds=(0., 1.), max_iter: int = 10_000
):
    """Shift the elements of `x` s.t. their mean is `target` while keeping them within `bounds`."""
    # Initial centroid
    mu = x.mean(dim=0)
    print("start", (mu - target).mean(), x.shape, target.shape, mu.shape) # each data point has a target mu, but there's only one mu

    for i in tqdm(range(max_iter)):
        x = x - (mu - target)
        x.clamp_(*bounds)

        mu = x.mean(dim=0)
        if torch.allclose(mu, target):
            print(f"Converged in {i} iterations")
            break

    print("Mean distance from target mean: ", (mu - target).mean())
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
        class_data[item["label"].item()].append(item["pixel_values"])
        i += 1
    
    # Calculate class mus
    mus = []
    for label in class_data.keys():
        class_data[label] = torch.stack(class_data[label])

        mu = class_data[label].float().mean(dim=0)
        mus.append(mu)

    # classes = list(range(len(original_labels)))

    # Mean shifted natural samples
    # shifted_samples = []
    # original_labels = []
    # for label, data in tqdm(class_data.items()):
    #     shift_classes = choices([c for c in classes if c != label], k=len(data))
    #     target_mus = torch.stack([mus[shift_class] for shift_class in shift_classes])
        
    #     shifted_samples.append(bounded_shift(data, target_mus, bounds=(0, 1)))
    #     original_labels.extend([label] * len(data))
    
    # data_dict = {
    #     'pixel_values': torch.cat(shifted_samples),
    #     'label': torch.tensor(original_labels)
    # }
    # Dataset.from_dict(data_dict).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/natural-{dataset}.hf')

    # Mean shifted maximum entropy samples
    shifted_samples = []
    original_labels = []
    for label in tqdm(class_data.keys()):
        mu = mus[label]
        print("mu", mu.isnan().sum())
        dd = DuryDistribution(mu.cpu().numpy(), start=0.1)
        samples = dd.sample(num_class_samples)

        # resample until there's no nans
        mask = np.logical_or(np.isnan(samples), np.isinf(samples))
        i = 0
        while mask.sum() > 0:
            i += 1
            print("i", i, mask.sum())
            dd = DuryDistribution(mu.cpu().numpy(), start=random.uniform(0, 1))
            resample = dd.sample(num_class_samples)
            samples[mask] = resample[mask]
            mask = np.logical_or(np.isnan(samples), np.isinf(samples))
        
        print("samples", np.sum(np.isinf(samples)), samples.min(), samples.max(), samples.mean())
        samples = torch.tensor(samples, dtype=torch.float)
        for i, mu in enumerate(mus):
            sample = samples[
                i * (len(samples) // len(mus)):
                i * (len(samples) // len(mus)) + len(samples) // len(mus)
            ]
            shifted_sample_batch = bounded_shift(sample, mu, bounds=(0., 1.))
            print("shifted_sample_batch", shifted_sample_batch.isnan().sum())
            shifted_samples.append(shifted_sample_batch)
            original_labels.extend([label] * len(sample))

            # Save it every time so I can at least get some data if it crashes
            data_dict = {
                'pixel_values': torch.cat(shifted_samples),
                'label': torch.tensor(original_labels),
            }
            Dataset.from_dict(data_dict).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/1-partial-max-entropy-{dataset}.hf')
        data_dict = {
            'pixel_values': torch.cat(shifted_samples),
            'label': torch.tensor(original_labels),
        }
        Dataset.from_dict(data_dict).save_to_disk(f'/mnt/ssd-1/lucia/shifted-data/1-max-entropy-{dataset}.hf')


if __name__ == "__main__":
    main()
