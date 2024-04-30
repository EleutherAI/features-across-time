import os
import random
from argparse import ArgumentParser
from pathlib import Path

from tqdm import tqdm
import safetensors
import numpy as np
import pandas as pd
import torch
import torchvision.transforms.functional as TF
from datasets import ClassLabel, DatasetDict, Features, Image, load_dataset, load_from_disk
from torch import Tensor, nn
from transformers.modeling_outputs import ModelOutput
from torch.utils.data import DataLoader

def load_state_dict_with_modified_keys(model, state_dict_path):
    loaded_state_dict = safetensors.torch.load_file(state_dict_path)

    modified_state_dict = {}
    for key, value in loaded_state_dict.items():
        modified_key = 'model.' + key
        modified_state_dict[modified_key] = value

    model.load_state_dict(modified_state_dict, strict=False)


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


def run_dataset(dataset_str: str, nets: list[str], seed: int, models_path: str):
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
    # print(f"Classes in '{dataset_str}': {labels}")

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["train"][0][img_col]
    c, (h, w) = len(example.mode), example.size
    # print(f"Image size: {h} x {w}")

    def preprocess(batch):
        return {
            "pixel_values": [TF.to_tensor(x) * 255 for x in batch[img_col]],
            "label": torch.tensor(batch[label_col]),
        }

    if val := ds.get("validation"):
        val = val.with_transform(preprocess)
    else:
        nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
        val = nontrain["train"].with_transform(preprocess)

    max_entropy_shifted_ds = load_from_disk(f'/mnt/ssd-1/lucia/shifted-data/partial-max-entropy-{dataset_str}')
    max_entropy_shifted_ds.set_format('torch', columns=['pixel_values','label'])
    natural_shifted_ds = load_from_disk(f'/mnt/ssd-1/lucia/shifted-data/natural-{dataset_str}')
    natural_shifted_ds.set_format('torch', columns=['pixel_values','label'])

    checkpoints = np.unique(2 ** np.arange(int(np.log2(1)), int(np.log2(65536)) + 1, dtype=int)).tolist()
    data_dicts = []

    with tqdm(total=len(nets) * len(checkpoints)) as pbar:
        for net in nets:
            for checkpoint in checkpoints:
                data_dicts.extend(
                    run_model(
                        max_entropy_shifted_ds,
                        natural_shifted_ds,
                        dataset_str,
                        net,
                        h,
                        len(labels),
                        seed,
                        models_path,
                        checkpoint
                    )
                )
                pbar.update(1)
    
    return data_dicts


def run_model(
    # Training data
    max_entropy_shifted_ds,
    natural_shifted_ds,
    ds_str: str,
    net_str: str,
    image_size: int,
    num_classes: int,
    seed: int,
    models_path: str,
    checkpoint: int
) -> list[dict]:  
    device = "cuda"
    model_path = os.path.join(models_path, ds_str)
    assert os.path.isdir(model_path)

    model_archs = [
        name for name in os.listdir(model_path) 
        if os.path.isdir(os.path.join(model_path, name)) and name.startswith(net_str)
    ]
    model_paths = [
        os.path.join(model_path, name, f'checkpoint-{checkpoint}') 
        for name in os.listdir(model_path) 
        if os.path.isdir(os.path.join(model_path, name)) and name.startswith(net_str)
    ]

    # print(f"Model sizes in {net_str}: {model_archs}")
    data = []
    for model_arch, model_path in zip(model_archs, model_paths):
        match model_arch.partition("-"):
            case ("convnext", _, arch):
                from transformers import ConvNextV2Config, ConvNextV2ForImageClassification
                match arch:
                    case "atto" | "":  # default
                        depths = [2, 2, 6, 2]
                        hidden_sizes = [40, 80, 160, 320]
                    case "femto":
                        depths = [2, 2, 6, 2]
                        hidden_sizes = [48, 96, 192, 384]
                    case "pico":
                        depths = [2, 2, 6, 2]
                        hidden_sizes = [64, 128, 256, 512]
                    case "nano":
                        depths = [2, 2, 8, 2]
                        hidden_sizes = [80, 160, 320, 640]
                    case "tiny":
                        depths = [3, 3, 9, 3]
                        hidden_sizes = [96, 192, 384, 768]
                    case other:
                        raise ValueError(f"Unknown ConvNeXt architecture {other}")

                cfg = ConvNextV2Config(
                    image_size=image_size,
                    depths=depths,
                    drop_path_rate=0.1,
                    hidden_sizes=hidden_sizes,
                    num_labels=num_classes,
                    # The default of 4 x 4 patches shrinks the image too aggressively for
                    # low-resolution images like CIFAR-10
                    patch_size=1,
                )
                model = ConvNextV2ForImageClassification.from_pretrained(model_path).to(device)
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
                load_state_dict_with_modified_keys(net, os.path.join(model_path, 'model.safetensors'))
                model = HfWrapper(net).to(device)

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
                load_state_dict_with_modified_keys(swin, os.path.join(model_path, 'model.safetensors'))
                model = HfWrapper(swin).to(device)
            case _:
                raise ValueError(f"Unknown model {model_arch}")

        # Collect data on model
        arch_data = {
            "step": checkpoint,
            "ds": ds_str,
            "net": net_str,
            "arch": model_arch
        }
        
        running_mean_loss = 0.0
        dataloader = DataLoader(max_entropy_shifted_ds, batch_size=512)
        for batch in dataloader:
            pixel_values = batch["pixel_values"].to(device) * 255
            print(pixel_values.isnan().sum())
            loss = model(pixel_values, batch["label"].to(device)).loss.item()
            running_mean_loss += (loss / len(dataloader))
        arch_data["maxent_shifted_loss"] = running_mean_loss
        
        # running_mean_loss = 0.0
        # dataloader = DataLoader(natural_shifted_ds, batch_size=512)
        # for batch in dataloader:
        #     loss = model(batch["pixel_values"].to(device) * 255, batch["label"].to(device)).loss.item()
        #     running_mean_loss += (loss / len(natural_shifted_ds))

        # arch_data["ds_shifted_loss"] = running_mean_loss
        # data.append(arch_data)

    return data


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "features-across-time"

    parser = ArgumentParser()
    parser.add_argument("--datasets", type=str, default=["cifar10", "svhn:cropped_digits", "mnist"], nargs="+")
    parser.add_argument(
        "--nets",
        type=str,
        choices=("convnext", "regnet", "swin", "vit"),
        nargs="+",
        default=["convnext", "regnet", "swin", "vit"]
    )
    parser.add_argument("--seed", type=int, default=0, help="Random seed")
    parser.add_argument("--checkpoints", type=str, default="/mnt/ssd-1/lucia/img-ckpts", help="Path to directory containing model checkpoints")
    args = parser.parse_args()

    data_dicts = []
    for dataset in args.datasets:
        data_dicts.extend(
            run_dataset(dataset, args.nets, args.seed, args.checkpoints)
        )
        df = pd.DataFrame(data_dicts)
        df.to_csv(f'vision-maxent-{dataset}.csv', index=False)
    
    df = pd.DataFrame(data_dicts)
    df.to_csv('vision-maxent.csv', index=False)

