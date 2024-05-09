import argparse
from dataclasses import dataclass
import random
import os

import numpy as np
import torch
from datasets import load_from_disk, load_dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl
)
import wandb

from script_utils.load_model import get_auto_tokenizer, get_auto_model
from script_utils.experiment import Experiment

torch.backends.cuda.matmul.allow_tf32 = True


def encode(example):
    example['labels'] = example['input_ids']
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
        gpu_rank = int(os.environ.get('LOCAL_RANK', '0'))
        if 'loss' in logs:
            print(f"{gpu_rank}: Step {state.global_step}: Training loss: {logs['loss']}")
        if 'eval_loss' in logs:
            print(f"{gpu_rank}: Step {state.global_step}: Evaluation loss: {logs['eval_loss']}")


def main(tmp_cache_path: str, data_path: str, seed=1):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

    experiments = [
        Experiment(
            team="EleutherAI", 
            model_name=model_name, 
            get_model=get_auto_model, 
            get_tokenizer=get_auto_tokenizer,
            batch_size=batch_size,
            training_arguments=TrainingArguments(
                output_dir=f"/mnt/ssd-1/lucia/{model_name}-es",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=4,
                num_train_epochs=1,
                evaluation_strategy="epoch",
                save_strategy="no",
                seed=seed,
                dataloader_num_workers=2,
                logging_steps=500,
                report_to="wandb",
            ),
            num_samples=1024,
            seq_len=2048, 
            d_vocab=50_277,
            steps=[],
            ngram_orders=[1, 2],
            eod_index=get_auto_tokenizer("EleutherAI", model_name).eos_token_id,
        )
        for model_name, batch_size in [
            ("pythia-14m", 4),
            # ("pythia-70m", 4),
            # ("pythia-160m", 4),
            # ("pythia-410m", 2),
            # ("pythia-1b", 2),
            # ("pythia-1.4b", 2),
            # ("pythia-2.8b", 2),
            # ("pythia-6.9b", 1),
            # ("pythia-12b", 1),
        ]
    ]

    gpu_ids = list(range(0, 8))
    for experiment in experiments:
        if not gpu_ids:
            gpu_ids = list(range(torch.cuda.device_count()))

        print(f"Parallelising over {len(gpu_ids)} GPUs...")

        dataset = load_from_disk(data_path).train_test_split(test_size=0.05)
        train_dataset = dataset["train"].with_transform(encode)
        test_dataset = dataset["test"].with_transform(encode)

        model = experiment.get_model(experiment.team, experiment.model_name, None, tmp_cache_path).float()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")

        if experiment.training_arguments.local_rank == 0 or experiment.training_arguments.local_rank == -1:
            wandb.init(project="pythia-es", entity="eleutherai")

        trainer = Trainer(model, 
                        experiment.training_arguments, 
                        train_dataset=train_dataset,
                        eval_dataset=test_dataset)
        trainer.add_callback(LossLoggingCallback())
        trainer.add_callback(LogSpacedCheckpoint())
        trainer.train()


# sudo -i
# cd /home/lucia/features-across-time && source /mnt/ssd-1/lucia/miniconda3/bin/activate /mnt/ssd-1/lucia/miniconda3/envs/3.10
# export PYTHONPATH="/home/lucia/features-across-time/src:$PYTHONPATH"
# accelerate launch --config_file scripts/config/default-config.yaml scripts/finetune.py

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
        default="/mnt/ssd-1/lucia/es_1b_full_tokenized.hf",
        help="Path to data",
    )
    parser.add_argument(
        "--seed",
        default=1
    )
    args = parser.parse_args()

    # train_dataset = load_dataset("allenai/c4", "es", split='train')
    # test_dataset = load_dataset("allenai/c4", "es", split='validation')
    # train_dataset = load_from_disk("/mnt/ssd-1/lucia/es_1b_full_tokenized.hf")
    # test_dataset = load_from_disk("/mnt/ssd-1/lucia/es_1b_full_tokenized.hf")

    main(args.tmp_cache_path, args.data_path, args.seed)