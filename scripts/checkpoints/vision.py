import os
import random
from argparse import ArgumentParser
from pathlib import Path
import yaml

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


def to_greyscale(ex):
    ex['pixel_values'] = ex['pixel_values'][0, :, :].unsqueeze(0).repeat(3, 1, 1).float()
    return ex


def preprocess(ex):
    # Seed everything
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ds_variations = {
        "maxent_shifted": load_from_disk(f'/mnt/ssd-1/lucia/shifted-data/max-entropy-{dataset_str}.hf'),
        "shifted": load_from_disk(f'/mnt/ssd-1/lucia/shifted-data/natural-{dataset_str}.hf'),
    }
    for ds_var in ds_variations.values():
        ds_var.set_format('torch', columns=['pixel_values','label'])

    if greyscale:
        for ds_name in ds_variations.keys():
            ds_variations[ds_name] = ds_variations[ds_name].map(to_greyscale)


    unique_labels = torch.unique(next(iter(ds_variations.values()))['label'])
    num_unique_labels = len(unique_labels)
    
    checkpoints = np.unique(2 ** np.arange(int(np.log2(1)), int(np.log2(65536)) + 1, dtype=int)).tolist()
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
                        seed,
                        models_path,
                        checkpoint,
                        batch_sizes
                    )
                )
                pbar.update(1)
    
    return data_dicts


def run_model(
    ds_variations,
    ds_str: str,
    net_str: str,
    num_classes: int,
    seed: int,
    models_path: str,
    checkpoint: int,
    batch_sizes: dict[str, int]
) -> list[dict]:  
    device = "cuda:0"
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

    data = []
    for model_arch, model_path in zip(model_archs, model_paths):
        match model_arch.partition("-"):
            case ("convnext", _, arch):
                from transformers import ConvNextV2ForImageClassification
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
                model = HfWrapper(net)
                model.load_state_dict(safetensors.torch.load_file(os.path.join(model_path, 'model.safetensors')))
                model.to(device)

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
                model.load_state_dict(safetensors.torch.load_file(os.path.join(model_path, 'model.safetensors')))
                model.to(device)
            case _:
                raise ValueError(f"Unknown model {model_arch}")

        arch_data = {
            "step": checkpoint,
            "ds": ds_str,
            "net": net_str,
            "arch": model_arch
        }
        batch_size = 512 if model_arch not in batch_sizes else batch_sizes[model_arch]

        for ds_name, ds in ds_variations.items():
            running_mean_loss = 0.0
            true_pred_count = 0.0
            dataloader = DataLoader(ds, batch_size=batch_size)
            for batch in dataloader:
                labels = batch["label"].to(device)
                output = model(batch["pixel_values"].to(device), labels)

                running_mean_loss += (output.loss.item() / len(dataloader))
                true_pred_count += output.logits.argmax(dim=-1).eq(labels).sum().item()
            del output
            arch_data[f"{ds_name}_loss"] = running_mean_loss
            arch_data[f"{ds_name}_accuracy"] = true_pred_count / len(ds)
        data.append(arch_data)

    return data


if __name__ == "__main__":
    os.environ["WANDB_PROJECT"] = "features-across-time"
    greyscale = True
    with open('/mnt/ssd-1/lucia/features-across-time/batch_sizes.yaml', 'r') as f:
        batch_sizes = yaml.safe_load(f)['A100']

    parser = ArgumentParser()
    parser.add_argument("--datasets", type=str, default=[
            "cifar10", 
            "svhn:cropped_digits", 
            "mnist", 
            # "evanarlian/imagenet_1k_resized_256", 
            # "fashion_mnist",
            # "cifarnet"
        ], nargs="+")
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

    with open('/scripts/batch_sizes.yaml', 'r') as f:
        batch_sizes = yaml.safe_load(f)['A100']
    
    data_dicts = []
    for dataset in args.datasets:
        data_dicts.extend(
            run_dataset(dataset, args.nets, args.seed, args.checkpoints, batch_sizes, greyscale)
        )
        df = pd.DataFrame(data_dicts)
        df.to_csv(Path('/mnt/ssd-1/lucia/24-05-15') / f'vision-{dataset}{"-greyscale" if greyscale else ""}.csv', index=False)
    
    df = pd.DataFrame(data_dicts)
    df.to_csv(Path('/mnt/ssd-1/lucia/24-05-15') / f'vision{"-greyscale" if greyscale else ""}.csv', index=False)