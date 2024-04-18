import os
import pickle
import random
from argparse import ArgumentParser
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
import safetensors
import numpy as np
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as TF
from concept_erasure import QuadraticEditor, QuadraticFitter
# from concept_erasure.quantile import QuantileNormalizer
from concept_erasure.utils import assert_type
from datasets import ClassLabel, Dataset, DatasetDict, Features, Image, load_dataset
from einops import rearrange
from torch import Tensor, nn, optim
from torch.distributions import MultivariateNormal
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
    get_cosine_schedule_with_warmup,
)
from transformers.modeling_outputs import ModelOutput
from scripts.script_utils.dury_distribution import DuryDistribution
from safetensors import safe_open
from safetensors.torch import load_model

# @dataclass
# class IndependentCoordinateSampler:
#     class_probs: Tensor
#     editor: QuantileNormalizer
#     size: int

#     def __getitem__(self, _: int) -> dict[str, Tensor]:
#         y = torch.multinomial(self.class_probs, 1).squeeze()
#         lut = self.editor.lut[y]

#         indices = torch.randint(0, lut.shape[-1], lut[..., 0].shape, device=lut.device)
#         x = lut.gather(-1, indices[..., None]).squeeze(-1)

#         return {
#             "pixel_values": x,
#             "label": y,
#         }

#     def __len__(self) -> int:
#         return self.size

def load_state_dict_with_modified_keys(model, state_dict_path):
    loaded_state_dict = safetensors.torch.load_file(state_dict_path)

    modified_state_dict = {}
    for key, value in loaded_state_dict.items():
        modified_key = 'model.' + key
        modified_state_dict[modified_key] = value

    model.load_state_dict(modified_state_dict, strict=False)


class GaussianMixture:
    def __init__(
        self,
        means: Tensor,
        covs: Tensor,
        class_probs: Tensor,
        size: int,
        shape: tuple[int, int, int] = (3, 32, 32),
        trf: Callable = lambda x: x,
    ):
        self.class_probs = class_probs
        self.dists = [MultivariateNormal(mean, cov) for mean, cov in zip(means, covs)]
        self.shape = shape
        self.size = size
        self.trf = trf

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        if idx >= self.size:
            raise IndexError(f"Index {idx} out of bounds for size {self.size}")

        y = torch.multinomial(self.class_probs, 1).squeeze()
        x = self.dists[y].sample().reshape(self.shape)
        return {
            "pixel_values": self.trf(x),
            "label": y,
        }

    def __len__(self) -> int:
        return self.size


@dataclass
class ConceptEditedDataset:
    class_probs: Tensor
    editor: QuadraticEditor
    X: Tensor
    Y: Tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], int(self.Y[idx])

        # Make sure we don't sample the correct class
        loo_probs = self.class_probs.clone()
        loo_probs[y] = 0
        target_y = torch.multinomial(loo_probs, 1).squeeze()

        x = self.editor.transport(x[None], y, int(target_y)).squeeze(0)
        return {
            "pixel_values": x,
            "label": target_y,
        }

    def __len__(self) -> int:
        return len(self.Y)


@dataclass
class QuantileNormalizedDataset:
    class_probs: Tensor
    # editor: QuantileNormalizer
    X: Tensor
    Y: Tensor

    def __getitem__(self, idx: int) -> dict[str, Tensor]:
        x, y = self.X[idx], self.Y[idx]

        # Make sure we don't sample the correct class
        loo_probs = self.class_probs.clone()
        loo_probs[y] = 0
        target_y = torch.multinomial(loo_probs, 1).squeeze()

        lut1 = self.editor.lut[y]
        lut2 = self.editor.lut[target_y]

        indices = torch.searchsorted(lut1, x[..., None]).clamp(0, lut1.shape[-1] - 1)
        x = lut2.gather(-1, indices).squeeze(-1)

        return {
            "pixel_values": x,
            "label": target_y,
        }

    def __len__(self) -> int:
        return len(self.Y)


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


def run_dataset(dataset_str: str, nets: list[str], seed: int, checkpoints: str):
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

    # Convert to RGB so we don't have to think about it
    ds = ds.map(lambda x: {img_col: x[img_col].convert("RGB")})

    # Infer the image size from the first image
    example = ds["train"][0][img_col]
    c, (h, w) = len(example.mode), example.size
    print(f"Image size: {h} x {w}")

    train_trf = T.Compose(
        [
            T.RandAugment(),
            T.RandomHorizontalFlip(),
            T.RandomCrop(h, padding=h // 8),
            T.ToTensor()
        ]
    )

    train = ds["train"].with_format("torch")
    X = assert_type(Tensor, train[img_col]).div(255)
    X = rearrange(X, "n h w c -> n c h w")
    Y = assert_type(Tensor, train[label_col])

    print("Computing statistics...")
    fitter = QuadraticFitter.fit(X.flatten(1).cuda(), Y.cuda())
    # normalizer = QuantileNormalizer(X, Y)
    print("Done.")

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

    class_probs = torch.bincount(Y).float()
    gaussian = GaussianMixture(
        fitter.mean_x.cpu(), fitter.sigma_xx.cpu(), class_probs, len(val), (c, h, w)
    )

    train = (
        ds["train"].with_transform(
            lambda batch: {
                "pixel_values": [train_trf(x) for x in batch[img_col]],
                "label": batch[label_col],
            },
        )
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

    # print(class_probs, normalizer, len(val))
    val_sets = {
        # "independent": IndependentCoordinateSampler(class_probs, normalizer, len(val)),
        "got": ConceptEditedDataset(class_probs, editor, X, Y),
        "gaussian": gaussian,
        "real": val,
        # "cqn": QuantileNormalizedDataset(class_probs, normalizer, X, Y),
    }
    for net in nets:
        for checkpoint in [1]:
            run_model(
                train,
                val_sets,
                test,
                dataset_str,
                net,
                h,
                len(labels),
                seed,
                checkpoints,
                checkpoint
            )


def run_model(
    # Training data
    train,
    val: dict[str, Dataset],
    test: Dataset | None,
    ds_str: str,
    net_str: str,
    image_size: int,
    num_classes: int,
    seed: int,
    checkpoints: str,
    checkpoint: int
):  
    # This is run over all models and datasets
    # For each checkpoint
    # Do optimal transport between class A and B
    # TODO do we need data for transporting each class to each other class?
    # There's no reason not to if it runs quickly.
    # But start with transporting each data to a random other class then update if it's fast

    model_path = os.path.join(checkpoints, ds_str)
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
    
    print(f"Model sizes in {net_str}: {model_archs}")

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
                model = ConvNextV2ForImageClassification.from_pretrained(model_path)
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
                model = HfWrapper(net)

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
                model = HfWrapper(swin)
            case _:
                raise ValueError(f"Unknown model {model_arch}")

        # raise NotImplementedError("Collect data on model")


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

    for dataset in args.datasets:
        run_dataset(dataset, args.nets, args.seed, args.checkpoints)


