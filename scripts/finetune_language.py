import argparse
import os
import random
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from datasets import load_from_disk
from script_utils.load_model import get_auto_model
from torch.optim import AdamW
from transformers import (
    Trainer,
    TrainerCallback,
    TrainerControl,
    TrainerState,
    TrainingArguments,
)
from transformers.optimization import get_cosine_with_min_lr_schedule_with_warmup
from transformers.trainer_utils import PREFIX_CHECKPOINT_DIR

import wandb

torch.backends.cuda.matmul.allow_tf32 = True


def encode(example):
    example["labels"] = example["input_ids"]
    return example


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

            control.should_evaluate = False
            control.should_save = True
            args.save_safetensors = False


class LossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        gpu_rank = int(os.environ.get("LOCAL_RANK", "0"))
        if "loss" in logs:
            print(
                f"{gpu_rank}: Step {state.global_step}: Training loss: {logs['loss']}"
            )
        if "eval_loss" in logs:
            print(
                f"{gpu_rank}: Step {state.global_step}: \
                    Evaluation loss: {logs['eval_loss']}"
            )


class SaveCompiledModelCallback(TrainerCallback):
    """torch.compile causes HF to save the model with additional state dict
    prefixes, so we save it directly to simplify loading."""

    def on_save(
        self,
        args: TrainingArguments,
        state: TrainerState,
        control: TrainerControl,
        **kwargs,
    ):
        if int(os.environ.get("LOCAL_RANK", "0")) == 0:
            checkpoint_folder = os.path.join(
                args.output_dir, f"{PREFIX_CHECKPOINT_DIR}-{state.global_step}"
            )

            state_dict_path = os.path.join(checkpoint_folder, "state_dict.pt")
            torch.save(kwargs["model"].state_dict(), state_dict_path)

            pytorch_model_path = os.path.join(checkpoint_folder, "pytorch_model.bin")
            if os.path.exists(pytorch_model_path):
                os.remove(pytorch_model_path)


def main(tmp_cache_path: str, data_path: Path, seed: int):
    print(f"Parallelising over {torch.cuda.device_count()} GPUs...")
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    ds = load_from_disk(str(data_path / "es_tokenized.hf")).train_test_split(
        test_size=0.05
    )
    train_ds = ds["train"].with_transform(encode)
    nontrain = ds["test"].train_test_split(train_size=1024, seed=seed)
    val_ds = nontrain["train"].with_transform(encode)
    test = nontrain["test"].with_transform(encode)

    unigram = load_from_disk(str(data_path / "1-gram-sequences.hf"))
    unigram.set_format(type="torch", columns=["input_ids"])
    bigram = load_from_disk(str(data_path / "2-gram-sequences.hf"))
    bigram.set_format(type="torch", columns=["input_ids"])
    val = {
        "1-gram": unigram.with_transform(encode),
        "2-gram": bigram.with_transform(encode),
        "real": val_ds,
    }

    for model_name, batch_size in [
        # ("pythia-14m", 16),
        # ("pythia-70m", 4),
        ("pythia-160m", 4),
        # ("pythia-410m", 2),
        # ("pythia-1b", 2),
        # ("pythia-1.4b", 2),
        # ("pythia-2.8b", 2),
        # ("pythia-6.9b", 1),
        # ("pythia-12b", 1),
    ]:
        model = get_auto_model("EleutherAI", model_name, None, tmp_cache_path).float()

        learning_rate = 1e-4
        optimizer = AdamW(
            model.parameters(),
            lr=learning_rate,
            betas=(0.9, 0.95),
            eps=1e-8,
            weight_decay=0.01,
        )
        lr_scheduler = get_cosine_with_min_lr_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(143_000 * 0.01),
            num_training_steps=143_000,
            min_lr=0.1 * learning_rate,
        )
        training_arguments = TrainingArguments(
            output_dir=f"ckpts/{model_name}-es-lr={learning_rate}",
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            num_train_epochs=1,
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="no",
            seed=seed,
            logging_steps=500,
            report_to="wandb",
            log_on_each_node=False,
            fp16=True,
        )

        if training_arguments.local_rank == 0 or training_arguments.local_rank == -1:
            wandb.init(project="pythia-es", entity="eleutherai")

        trainer = Trainer(
            model,
            training_arguments,
            optimizers=(optimizer, lr_scheduler),
            train_dataset=train_ds,
            callbacks=[
                LogSpacedCheckpoint(),
                LossLoggingCallback(),
                SaveCompiledModelCallback(),
            ],
            eval_dataset=val,
        )
        trainer.train()
        trainer.evaluate(test)


# accelerate launch --config_file scripts/config/default-config.yaml scripts/finetune_language.py
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache (repeatedly cleared to free disk space)",
    )
    parser.add_argument(
        "--data_path",
        default="data/es",
        help="Path to datasets",
    )
    parser.add_argument("--seed", default=0)
    args = parser.parse_args()

    main(args.tmp_cache_path, Path(args.data_path), args.seed)
