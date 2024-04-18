import argparse

import torch

from dataclasses import dataclass

from datasets import load_from_disk, load_dataset
from transformers import (
    Trainer, 
    TrainingArguments, 
    TrainerCallback, 
    TrainerState, 
    TrainerControl
)

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


def main(tmp_cache_path: str):
    experiments = [
        Experiment(
            team="EleutherAI", 
            model_name=model_name, 
            get_model=get_auto_model, 
            get_tokenizer=get_auto_tokenizer,
            batch_size = 2,
            training_arguments=TrainingArguments(
                output_dir="/mnt/ssd-1/lucia/",
                per_device_train_batch_size=batch_size,
                per_device_eval_batch_size=4,
                num_train_epochs=1,
                evaluation_strategy="epoch",
                save_strategy="no",
                seed=1,
                dataloader_num_workers=2
            ),
            num_samples=1024,
            seq_len=2048, 
            d_vocab=50_277,
            steps=[],
            ngram_orders=[1, 2],
            eod_index=get_auto_tokenizer("EleutherAI", model_name).eos_token_id,
        )
        for model_name, batch_size in [
            # ("pythia-14m", 4),
            # ("pythia-70m", 4),
            ("pythia-160m", 2),
            # ("pythia-410m", 4),
            # ("pythia-1b", 4),
            # ("pythia-1.4b", 4),
            # ("pythia-2.8b", 4),
            # ("pythia-6.9b", 1),
            # ("pythia-12b", 1),
        ]
    ]

    gpu_ids = None
    for experiment in experiments:

        if not gpu_ids:
            gpu_ids = list(range(torch.cuda.device_count()))

        print(f"Parallelising over {len(gpu_ids)} GPUs...")

        # train_dataset = load_dataset("allenai/c4", "es", split='train')
        # test_dataset = load_dataset("allenai/c4", "es", split='validation')
        train_dataset = load_from_disk("/mnt/ssd-1/lucia/es_1b_full_tokenized.hf")
        test_dataset = load_from_disk("/mnt/ssd-1/lucia/es_1b_full_tokenized.hf")

        train_dataset = train_dataset.with_transform(encode)
        test_dataset = test_dataset.with_transform(encode)

        model = experiment.get_model(experiment.team, experiment.model_name, None, tmp_cache_path).float()

        trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
        print(f"Number of trainable parameters: {trainable_params}")
        
        trainer = Trainer(model, 
                        experiment.training_arguments, 
                        train_dataset=train_dataset, 
                        eval_dataset=test_dataset)

        trainer.add_callback(LogSpacedCheckpoint())
        trainer.train()


# Run with `accelerate launch <script_name>.py`
if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    parser.add_argument(
        "--tmp_cache_path",
        default=".cache",
        help="Path to cache (repeatedly cleared to free disk space)",
    )
    args = parser.parse_args()
    main(args.tmp_cache_path)