import os
from pathlib import Path

import torch
from transformers import GPTNeoXForCausalLM


def get_es_finetune(
    team: str, model_name: str, step: int | None, cache_dir: str, device: str = "cuda"
):
    state_dict_path = (
        Path("ckpts") / f"{model_name}-es" / f"checkpoint-{step}" / "state_dict.pt"
    )
    assert os.path.isfile(
        state_dict_path
    ), f"Local state dict {state_dict_path} does not exist"

    model = GPTNeoXForCausalLM.from_pretrained(
        f"{team}/{model_name}", torch_dtype="auto", cache_dir=cache_dir
    ).to(device)
    model.load_state_dict(torch.load(state_dict_path))
    return model
