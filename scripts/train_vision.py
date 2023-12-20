from argparse import ArgumentParser
from dataclasses import dataclass
from typing import Callable

import numpy as np
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from concept_erasure import QuadraticEditor, QuadraticFitter, groupby
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from einops import rearrange
from torch import Tensor
from torch.distributions import MultivariateNormal
from tqdm.auto import tqdm
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)


class FakeImages:
    def __init__(
        self,
        root: str,
        names: list[str],
        trf: Callable = lambda x: x,
    ):
        class_list = sorted((name, torch.load(f"{root}/{name}.pt")) for name in names)
        self.data = torch.cat([x for _, x in class_list])
        self.labels = torch.cat(
            [
                torch.full([len(x)], i, device=x.device)
                for i, (_, x) in enumerate(class_list)
            ]
        )
        self.trf = trf

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x = rearrange(self.data[idx], "(h w c) -> c h w", c=3, h=32, w=32)
        return {
            "pixel_values": self.trf(x),
            "label": self.labels[idx],
        }

    def __len__(self) -> int:
        return len(self.data)


class GaussianMixture:
    def __init__(
        self,
        means: Tensor,
        covs: Tensor,
        size: int,
        trf: Callable = lambda x: x,
    ):
        self.dists = [MultivariateNormal(mean, cov) for mean, cov in zip(means, covs)]
        self.size = size
        self.trf = trf

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        y = idx % len(self.dists)
        x = rearrange(self.dists[y].sample(), "(h w c) -> c h w", c=3, h=32, w=32)
        return {
            "pixel_values": self.trf(x),
            "label": torch.tensor(y, device=x.device),
        }

    def __len__(self) -> int:
        return self.size


@dataclass
class ConceptEditedDataset:
    dataset: Dataset
    editor: QuadraticEditor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        idx, target_y = divmod(idx, len(self.editor.class_means))

        record = self.dataset[idx]
        x, y = record["pixel_values"], record["label"]

        x = self.editor.transport(x.view(1, -1), y, target_y).flatten()
        x = rearrange(x, "(h w c) -> c h w", c=3, h=32, w=32)

        return {
            "pixel_values": x,
            "label": torch.tensor(target_y, device=x.device),
        }

    def __len__(self) -> int:
        return len(self.dataset)


@dataclass
class LogSpacedCheckpoint(TrainerCallback):
    """Save checkpoints at log-spaced intervals"""

    base: float = 2.0
    next: int = 1

    def on_step_end(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if state.global_step >= self.next:
            self.next = round(self.next * self.base)

            control.should_evaluate = True
            control.should_save = True


def infer_columns(feats: Features) -> tuple[str, str]:
    # Infer the appropriate columns by their types
    img_cols = [k for k in feats if isinstance(feats[k], Image)]
    label_cols = [k for k in feats if isinstance(feats[k], ClassLabel)]

    assert len(img_cols) == 1, f"Found {len(img_cols)} image columns"
    assert len(label_cols) == 1, f"Found {len(label_cols)} label columns"

    return img_cols[0], label_cols[0]


def run_dataset(dataset_str: str, nets: list[str], train_on_fake: bool):
    train_trf = T.Compose(
        [
            T.RandAugment() if not train_on_fake else T.Lambda(lambda x: x),
            T.RandomHorizontalFlip(),
            T.RandomCrop(32, padding=4),
            T.ToTensor() if not train_on_fake else T.Lambda(lambda x: x),
        ]
    )
    nontrain_trf = T.ToTensor()

    # Allow specifying load_dataset("svhn", "cropped_digits") as "svhn:cropped_digits"
    # We don't use the slash because it's a valid character in a dataset name
    path, _, name = dataset_str.partition(":")
    ds = load_dataset(path, name or None)
    assert isinstance(ds, DatasetDict)

    # Infer columns and class labels
    img_col, label_col = infer_columns(ds["train"].features)
    labels = ds["train"].features[label_col].names
    print(f"Classes in '{dataset_str}': {labels}")

    # Infer the image size from the first image
    h, w = ds["train"][0][img_col].size
    print(f"Image size: {h} x {w}")

    # Build the Q-LEACE editor
    fitter = QuadraticFitter(h * w * 3, len(labels), device="cuda")
    train = ds["train"].with_format("torch")

    for x, y in zip(tqdm(train[img_col]), train[label_col]):
        fitter.update_single(x.view(1, -1).cuda().div(255), int(y))

    test = ds["test"].with_transform(
        lambda batch: {
            "pixel_values": [nontrain_trf(x) for x in batch[img_col]],
            "label": batch[label_col],
        },
    )

    if val := ds.get("validation"):
        val = val.with_transform(
            lambda batch: {
                "pixel_values": [nontrain_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
            },
        )
    else:
        nontrain = test.train_test_split(train_size=1024, seed=0)
        val, test = nontrain["train"], nontrain["test"]

    gaussian = FakeImages("/root/cifar-fake", labels, train_trf)
    # gaussian = GaussianMixture(fitter.mean_x, fitter.sigma_xx, len(val))
    print("Computing optimal transport maps...")

    train = (
        ds["train"].with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
            },
        )
        if not train_on_fake
        else gaussian
    )

    val_sets = {
        "real": val,
        "edited": ConceptEditedDataset(val, fitter.editor("cpu")),
        "gaussian": gaussian,
    }
    for net in nets:
        run_model(train, val_sets, test, net, len(labels))


def run_model(train, val: dict[str, Dataset], test: Dataset, net_str: str, k: int):
    match net_str:
        case "convnext":
            from transformers import ConvNextV2Config, ConvNextV2ForImageClassification

            cfg = ConvNextV2Config(
                image_size=32,
                # Atto architecture
                depths=[2, 2, 6, 2],
                drop_path_rate=0.1,
                hidden_sizes=[40, 80, 160, 320],
                num_labels=k,
                # The default of 4 x 4 patches shrinks the image too aggressively for
                # low-resolution images like CIFAR-10
                patch_size=1,
            )
            model = ConvNextV2ForImageClassification(cfg)
        case "regnet":
            from transformers import RegNetConfig, RegNetForImageClassification

            cfg = RegNetConfig(
                image_size=32,
                hidden_sizes=[40, 80, 160, 320],
                num_labels=k,
            )
            model = RegNetForImageClassification(cfg)
        case "resnet":
            from transformers import ResNetConfig, ResNetForImageClassification

            cfg = ResNetConfig(
                image_size=32,
                hidden_sizes=[40, 80, 160, 320],
                num_labels=k,
            )
            model = ResNetForImageClassification(cfg)
        case "vit":
            from transformers import ViTConfig, ViTForImageClassification

            cfg = ViTConfig(
                image_size=32,
                hidden_sizes=[40, 80, 160, 320],
                num_labels=k,
            )
            model = ViTForImageClassification(cfg)
        case _:
            raise ValueError(f"Unknown net {net_str}")

    names = [
        "airplane",
        "automobile",
        "bird",
        "cat",
        "deer",
        "dog",
        "frog",
        "horse",
        "ship",
        "truck",
    ]
    trainer = Trainer(
        model,
        args=TrainingArguments(
            output_dir=f"runs/{val['real'].builder_name}/{net_str}",
            adam_beta2=0.95,
            fp16=True,
            # Don't pin memory for GaussianMixture because it's already on the GPU
            dataloader_pin_memory=not isinstance(train, (GaussianMixture, FakeImages)),
            learning_rate=0.005,
            lr_scheduler_type="cosine",
            max_steps=2**16,
            per_device_train_batch_size=128,
            remove_unused_columns=False,
            save_strategy="no",
            tf32=True,
            warmup_steps=2000,
            weight_decay=0.05,
        ),
        callbacks=[LogSpacedCheckpoint()],
        compute_metrics=lambda x: {
            "acc": np.mean(x.label_ids == np.argmax(x.predictions, axis=-1))
        }
        | {
            "loss_"
            + names[i]: F.cross_entropy(
                grp, grp.new_full((len(grp),), i, dtype=torch.long)
            ).item()
            for i, grp in groupby(
                torch.from_numpy(x.predictions), torch.from_numpy(x.label_ids)
            )
        },
        train_dataset=train,
        eval_dataset=val,
    )
    trainer.train()
    trainer.evaluate(test)


if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument("--datasets", type=str, default=["cifar10"], nargs="+")
    parser.add_argument(
        "--nets",
        type=str,
        choices=("convnext", "regnet", "resnet", "swin", "vit"),
        nargs="+",
    )
    parser.add_argument(
        "--train-on-fake",
        action="store_true",
    )
    args = parser.parse_args()

    for dataset in args.datasets:
        run_dataset(dataset, args.nets, args.train_on_fake)
