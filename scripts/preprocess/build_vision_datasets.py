import os
import random
from argparse import ArgumentParser
from collections import defaultdict
from pathlib import Path

import numpy as np
import plotly.express as px
import torch
import torchvision.transforms.functional as TF
from concept_erasure.utils import assert_type
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from einops import rearrange
from torch import Tensor
from tqdm import tqdm

from scripts.script_utils.dury_distribution import DuryDistribution
from scripts.script_utils.truncated_normal import truncated_normal


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def bounded_shift(x: Tensor, target: Tensor, bounds=(0.0, 1.0), max_iter: int = 10_000):
    """Shift the elements of `x` s.t. their mean is `target`
    while keeping them within `bounds`."""
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


def to_psd(sigma):
    return sigma.T @ sigma / len(sigma)


def truncated_normal_ds_with_moments(
    n: int,
    class_moments: list[tuple],
    seed: int,
    data_path: Path,
    ds_name: str,
    sample_item: Tensor,
):
    X = []
    Y = []

    for i, (mu, sigma) in enumerate(class_moments):
        data = truncated_normal(n, mu, sigma, seed=seed)
        X.append(data.reshape((n, *sample_item.shape)))
        Y.extend([i] * n)

        fig = px.histogram(X[-1].cpu()[:, 0, 0, 0])
        fig.write_image(f"mu-class-{i}-{ds_name}.png", format="png")

    data_dict = {
        "pixel_values": torch.cat(X),
        "label": torch.tensor(Y),
    }
    ds = Dataset.from_dict(data_dict).shuffle(seed)
    ds.save_to_disk(str(data_path / ds_name))
    return ds


def build_from_dataset(dataset_str: str, output_path: Path, seed: int):
    os.makedirs(output_path, exist_ok=True)

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

    class_data = defaultdict(list)
    for target in range(len(X)):
        class_data[Y[target].item()].append(X[target])

    mus = {
        key: torch.stack(data).float().mean(dim=0) for key, data in class_data.items()
    }

    def high_var_ds():
        device = "cuda:0"
        num_samples = 50_000  # len(ds["train"])
        # seed = 1
        sample_item = class_data[0][0].permute(1, 0, 2)
        sample_sigma = torch.stack([sample_item, sample_item]).flatten(1, 3).T.cov()

        # print("Generating high variance difference classes with no mean differences...")
        # mu = torch.full_like(sample_item.flatten(), 0.5, device=device)
        # class_moments = [
        #     (
        #         mu,
        #         torch.full_like(sample_sigma, 0.5, device=device),
        #     ),
        #     (
        #         mu,
        #         torch.full_like(sample_sigma, 0.001, device=device),
        #     )
        # ]
        # truncated_normal_ds_with_moments(
        #     num_samples,
        #     class_moments,
        #     seed,
        #     output_path,
        #     f'var_diff.hf',
        #     sample_item
        # )

        # print("Generating high mean difference classes with no variance differences...")
        # sigma = torch.full_like(sample_sigma, 0.5, device=device)
        # class_moments = [
        #     (
        #         torch.full_like(sample_item.flatten(), 0.2, device=device),
        #         sigma
        #     ),
        #     (
        #         torch.full_like(sample_item.flatten(), 0.8, device=device),
        #         sigma
        #     )
        # ]
        # truncated_normal_ds_with_moments(
        #     num_samples,
        #     class_moments,
        #     seed,
        #     output_path,
        #     f'mean_diff.hf',
        #     sample_item
        # )

        # print("Generating high_mu_low_sigma...")
        # class_moments = [
        #     (
        #         torch.full_like(sample_item.flatten(), 0.2, device=device),
        #         torch.full_like(sample_sigma, 0.004, device=device),
        #     ),
        #     (
        #         torch.full_like(sample_item.flatten(), 0.8, device=device),
        #         torch.full_like(sample_sigma, 0.001, device=device),
        #     )
        # ]
        # truncated_normal_ds_with_moments(
        #     num_samples,
        #     class_moments,
        #     seed,
        #     output_path,
        #     f'high_mu_low_sigma.hf',
        #     sample_item
        # )

        print("Generating low_mu_high_sigma...")
        # Each pixel has a slightly different (mu, sigma) to work around determinism in the sampler
        mu = 0.44
        class_moments = [
            (
                mu + torch.rand_like(sample_item.flatten(), device=device) * 0.001,
                to_psd(0.5 + torch.rand_like(sample_sigma, device=device) * 0.001),
            ),
            (
                mu + torch.rand_like(sample_item.flatten(), device=device) * 0.001,
                to_psd(0.001 + torch.rand_like(sample_sigma, device=device) * 0.001),
            ),
        ]
        truncated_normal_ds_with_moments(
            num_samples,
            class_moments,
            seed,
            output_path,
            "high_sigma_diff.hf",
            sample_item,
        )

    high_var_ds()

    def mean_shifted_ds():
        print("Generating mean shifted data...")
        X = []
        Y = []
        prev_Y = []
        for label, data in tqdm(class_data.items()):
            targets = {i: mu for i, mu in mus.items() if i != label}

            for target, mu in targets.items():
                shifted = bounded_shift(torch.stack(data), mu, bounds=(0.0, 1.0))
                Y.extend([target] * len(data))
                prev_Y.extend([label] * len(data))
                X.append(shifted)

        data_dict = {
            "pixel_values": torch.cat(X),
            "original": torch.tensor(prev_Y),
            "label": torch.tensor(Y),
        }
        Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(
            str(output_path / f'shifted-{dataset_str.replace("/", "--")}.hf')
        )

    # mean_shifted_ds()

    def dury_ds():
        print("Generating Dury distribution maximum entropy data...")
        X = []
        Y = []
        for label, original_data in tqdm(class_data.items()):
            mu = mus[label].numpy()
            dd = DuryDistribution(mu)
            data = dd.sample(len(original_data))
            X.append(torch.tensor(data, dtype=torch.float))
            Y.extend([label] * len(original_data))

        data_dict = {"pixel_values": torch.cat(X), "label": torch.tensor(Y)}
        Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(
            str(output_path / f'dury-{dataset_str.replace("/", "--")}.hf')
        )

    # dury_ds()

    def truncated_normal_ds():
        print("Generating truncated normal maximum entropy data...")
        # Use train dataset to stabilize sampling
        train_data = ds["train"].with_transform(preprocess)
        with train_data.formatted_as("torch"):
            X = assert_type(Tensor, train_data[img_col]).div(255)
            X = rearrange(X, "n h w c -> n c h w")
            Y = assert_type(Tensor, train_data[label_col])

        train_class_data = defaultdict(list)
        for target in range(len(X)):
            train_class_data[Y[target].item()].append(X[target])

        X = []
        Y = []
        for label, data in tqdm(train_class_data.items()):
            sigma = torch.stack(data).float().flatten(1, 3).T.cov()
            sample = truncated_normal(
                len(data),
                torch.stack(data).float().mean(dim=0).flatten(),
                sigma,
                seed=seed,
            )
            X.append(sample.reshape(len(data), *data[0].shape))
            Y.extend([label] * len(data))

        data_dict = {"pixel_values": torch.cat(X), "label": torch.tensor(Y)}
        Dataset.from_dict(data_dict).shuffle(seed).save_to_disk(
            str(output_path / f'truncated-normal-{dataset_str.replace("/", "--")}.hf')
        )

    # truncated_normal_ds()


def main():
    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default=[
            "cifar10",
            # "svhn:cropped_digits",
            # "mnist",
            # "fashion_mnist",
            # "EleutherAI/cifarnet",
        ],
        nargs="+",
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--output",
        type=str,
        default=Path.cwd() / "data",
        help="Path to directory containing output data",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        build_from_dataset(dataset, Path(args.output), args.seed)


if __name__ == "__main__":
    main()
