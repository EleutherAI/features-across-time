from dataclasses import dataclass
from typing import Callable, Any
import math

import torch
import pandas as pd
import torch.multiprocessing as mp


@dataclass
class Experiment:
    """Any configuration required by any worker."""
    num_samples: int
    batch_size: int
    seq_len: int
    team: str
    model_name: str
    get_model: Callable[[str, str, int | None, str], Any]
    get_tokenizer: Callable[[], Any]
    d_vocab: int
    steps: list[int]
    ngram_orders: list[int]
    
    epochs: int | None
    lr: float | None
    gamma: float | None
    seed: int | None
    test_batch_size: int | None
    save_model: bool = False

    eod_index: int | None = None


def run_checkpoint_experiment_workers(
    experiment: Experiment, 
    worker: Callable[[int, Experiment, str, str, str, list[str]], pd.DataFrame],
    ngram_path: str, 
    pile_path: str, 
    tmp_cache_path: str,
    gpu_ids: list[int] = None
):
    if not gpu_ids:
        gpu_ids = list(range(torch.cuda.device_count()))

    max_steps_per_chunk = math.ceil(len(experiment.steps) / len(gpu_ids))
    
    step_indices = [
        experiment.steps[i : i + max_steps_per_chunk]
        for i in range(0, len(experiment.steps), max_steps_per_chunk)
    ]
    print(step_indices)

    args = [
        (
            gpu_id,
            experiment,
            ngram_path,
            pile_path,
            tmp_cache_path,
            step_indices[i],
        )
        for i, gpu_id in enumerate(gpu_ids[:len(step_indices)])
    ]
    print(f"Parallelising over {len(step_indices)} GPUs...")
    with mp.Pool(len(step_indices)) as pool:
        dfs = pool.starmap(worker, args)

    return pd.concat(dfs)