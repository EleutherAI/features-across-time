import os
import pickle
import random
from argparse import ArgumentParser
from pathlib import Path

import numpy as np
import pandas as pd
import safetensors
import torch
import torchvision.transforms.functional as TF
from concept_erasure import QuadraticFitter
from concept_erasure.quantile import QuantileNormalizer
from concept_erasure.utils import assert_type
from datasets import (
    ClassLabel,
    DatasetDict,
    Features,
    Image,
    load_dataset,
    load_from_disk,
)
from einops import rearrange
from torch import Tensor, nn
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers.modeling_outputs import ModelOutput

from scripts.train_vision import (
    ConceptEditedDataset,
    GaussianMixture,
    IndependentCoordinateSampler,
    QuantileNormalizedDataset,
)


class HfWrapper(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, pixel_values: Tensor, labels: Tensor | None = None):
        logits = self.model(pixel_values)
        loss = (
            nn.functional.cross_entropy(logits, labels) if labels is not None else None
        )
        return ModelOutput(logits=logits, loss=loss)


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def to_grayscale(ex):
    ex["pixel_values"] = ex["pixel_values"][:1, :, :].repeat(3, 1, 1).float()
    return ex


def run_dataset(
    dataset_str: str, nets: list[str], seed: int, models_path: str, data_path: str
):
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    img_col, label_col = infer_columns(ds["train"].features)
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["train"][0][img_col]
    c, (h, w) = len(example.mode), example.size

    train = ds["train"].with_format("torch")
    X = assert_type(Tensor, train[img_col]).div(255)
    X = rearrange(X, "n h w c -> n c h w")
    Y = assert_type(Tensor, train[label_col])

    print("Computing statistics...")
    fitter = QuadraticFitter.fit(X.flatten(1), Y)
    normalizer = QuantileNormalizer(X, Y)
    print("Done.")

    def preprocess(ex):
        ex["pixel_values"] = TF.to_tensor(ex[img_col])
        ex["label"] = torch.tensor(ex[label_col])
        return ex

    if val := ds.get("validation"):
        val = val.map(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].map(preprocess)

    class_probs = torch.bincount(Y).float()
    gaussian = GaussianMixture(
        fitter.mean_x.cpu(), fitter.sigma_xx.cpu(), class_probs, len(val), (c, h, w)
    )

    cache = Path.cwd() / "editor-cache" / f"{dataset_str}.pkl"
    if cache.exists():
        with open(cache, "rb") as f:
            editor = pickle.load(f)
    else:
        print("Computing optimal transport maps...")

        editor = fitter.editor("cpu")
        cache.parent.mkdir(exist_ok=True)

        with open(cache, "wb") as f:
            pickle.dump(editor, f)

    with val.formatted_as("torch"):
        X = assert_type(Tensor, val[img_col]).div(255)
        X = rearrange(X, "n h w c -> n c h w")
        Y = assert_type(Tensor, val[label_col])

    dataset_file_str = dataset_str.replace("/", "--")
    truncated_normal = load_from_disk(
        data_path / f"truncated-normal-{dataset_file_str}.hf"
    ).select(range(len(val)))
    
    ds_variations = {
        "maxent": load_from_disk(data_path / f"dury-{dataset_file_str}.hf"),
        "shifted": load_from_disk(data_path / f"shifted-{dataset_file_str}.hf"),
        "truncated_normal": truncated_normal,
        "real": val,
        "independent": IndependentCoordinateSampler(class_probs, normalizer, len(val)),
        "got": ConceptEditedDataset(class_probs, editor, X, Y),
        "gaussian": gaussian,
        "cqn": QuantileNormalizedDataset(class_probs, normalizer, X, Y),
    }

    for ds_name in ["maxent", "shifted", "real", "truncated_normal"]:
        ds_variations[ds_name].set_format("torch", columns=["pixel_values", label_col])

    if dataset_str == "mnist" or dataset_str == "fashion_mnist":
        for ds_name in ["maxent", "shifted", "truncated_normal"]:
            ds_variations[ds_name] = ds_variations[ds_name].map(to_grayscale)

    num_unique_labels = len(torch.unique(next(iter(ds_variations.values()))[label_col]))
    checkpoints = np.unique(
        2 ** np.arange(int(np.log2(1)), int(np.log2(65536)) + 1, dtype=int)
    ).tolist()
    data_dicts = []

    with tqdm(total=len(nets) * len(checkpoints)) as pbar:
        for net in nets:
            for checkpoint in checkpoints:
                data_dicts.extend(
                    run_model(
                        ds_variations,
                        dataset_str,
                        net,
                        num_unique_labels,
                        models_path,
                        checkpoint,
                    )
                )
                pbar.update(1)

    return data_dicts


def run_model(
    ds_variations,
    ds_str: str,
    net_str: str,
    num_classes: int,
    models_path: str,
    checkpoint: int,
) -> list[dict]:
    model_path = os.path.join(models_path, ds_str)
    assert os.path.isdir(model_path)

    model_archs = [
        name
        for name in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, name)) and name.startswith(net_str)
    ]
    model_paths = [
        os.path.join(model_path, name, f"checkpoint-{checkpoint}")
        for name in os.listdir(model_path)
        if os.path.isdir(os.path.join(model_path, name)) and name.startswith(net_str)
    ]

    data = []
    for model_arch, model_path in zip(model_archs, model_paths):
        match model_arch.partition("-"):
            case ("convnext", _, arch):
                from transformers import ConvNextV2ForImageClassification

                model = ConvNextV2ForImageClassification.from_pretrained(
                    model_path
                ).cuda()
            case ("regnet", _, arch):
                from torchvision.models import (
                    regnet_y_1_6gf,
                    regnet_y_3_2gf,
                    regnet_y_400mf,
                    regnet_y_800mf,
                )

                match arch:
                    case "400mf":
                        net = regnet_y_400mf(num_classes=num_classes)
                    case "800mf":
                        net = regnet_y_800mf(num_classes=num_classes)
                    case "1.6gf":
                        net = regnet_y_1_6gf(num_classes=num_classes)
                    case "3.2gf":
                        net = regnet_y_3_2gf(num_classes=num_classes)
                    case other:
                        print(other, type(other))
                        raise ValueError(f"Unknown RegNet architecture {other}")

                net.stem[0].stride = (1, 1)  # type: ignore
                model = HfWrapper(net)
                state_dict_path = os.path.join(model_path, "model.safetensors")
                model.load_state_dict(safetensors.torch.load_file(state_dict_path))
                model.cuda()

            case ("swin", _, arch):
                from torchvision.models.swin_transformer import (
                    PatchMergingV2,
                    SwinTransformer,
                    SwinTransformerBlockV2,
                )

                match arch:
                    case "atto":
                        num_heads = [2, 4, 8, 16]
                        embed_dim = 40
                    case "femto":
                        num_heads = [2, 4, 8, 16]
                        embed_dim = 48
                    case "pico":
                        num_heads = [2, 4, 8, 16]
                        embed_dim = 64
                    case "nano":
                        num_heads = [2, 4, 8, 16]
                        embed_dim = 80
                    case "tiny" | "":  # default
                        num_heads = [3, 6, 12, 24]
                        embed_dim = 96
                    case other:
                        raise ValueError(f"Unknown Swin architecture {other}")

                # Tiny architecture with 2 x 2 patches
                swin = SwinTransformer(
                    patch_size=[2, 2],
                    embed_dim=embed_dim,
                    depths=[2, 2, 6, 2],
                    num_heads=num_heads,
                    window_size=[7, 7],
                    num_classes=num_classes,
                    stochastic_depth_prob=0.2,
                    block=SwinTransformerBlockV2,
                    downsample_layer=PatchMergingV2,
                )
                model = HfWrapper(swin)
                state_dict_path = os.path.join(model_path, "model.safetensors")
                model.load_state_dict(safetensors.torch.load_file(state_dict_path))
                model.cuda()
            case _:
                raise ValueError(f"Unknown model {model_arch}")

        arch_data = {
            "step": checkpoint,
            "ds": ds_str,
            "net": net_str,
            "arch": model_arch,
        }

        for ds_name, ds in ds_variations.items():
            running_mean_loss = 0.0
            true_pred_count = 0.0
            dataloader = DataLoader(ds, batch_size=512, drop_last=True)
            for batch in dataloader:
                labels = batch["label"].cuda()
                output = model(batch["pixel_values"].cuda(), labels)

                running_mean_loss += output.loss.item() / len(dataloader)
                true_pred_count += output.logits.argmax(dim=-1).eq(labels).sum().item()

            arch_data[f"{ds_name}_loss"] = running_mean_loss
            arch_data[f"{ds_name}_accuracy"] = true_pred_count / len(ds)
            print(ds_name, model_arch, running_mean_loss, true_pred_count / len(ds))
        data.append(arch_data)

    return data


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "features-across-time"

    parser = ArgumentParser()
    parser.add_argument(
        "--datasets",
        type=str,
        default=[
            "cifar10",
            "svhn:cropped_digits",
            "mnist",
            "fashion_mnist",
            "EleutherAI/cifarnet",
        ],
        nargs="+",
    )
    parser.add_argument(
        "--nets",
        type=str,
        choices=("convnext", "regnet", "swin", "vit"),
        nargs="+",
        default=["convnext", "regnet", "swin", "vit"],
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument(
        "--data",
        type=str,
        default=Path.cwd() / "vision-data",
        help="Path to directory containing local datasets",
    )
    parser.add_argument(
        "--checkpoints",
        type=str,
        default=Path.cwd() / "img-ckpts",
        help="Path to directory containing model checkpoints",
    )
    parser.add_argument(
        "--sink",
        type=str,
        default=Path.cwd() / "24-05-23",
        help="Path to directory containing output data",
    )
    args = parser.parse_args()

    data_dicts = []
    for dataset in args.datasets:
        data_dict = run_dataset(
            dataset, args.nets, args.seed, args.checkpoints, args.data
        )
        df = pd.DataFrame(data_dict)

        dataset_file_str = dataset.replace("/", "--")
        df.to_csv(
            Path(args.sink) / f"vision-{args.nets}-{dataset_file_str}.csv", index=False
        )
