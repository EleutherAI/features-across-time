import os
from pathlib import Path

import torch
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    GPTNeoXForCausalLM,
    LlamaTokenizer,
)


def get_auto_tokenizer(team: str, model_name: str):
    return AutoTokenizer.from_pretrained(f"{team}/{model_name}")


def get_amber_tokenizer(team: str, model_name: str):
    return LlamaTokenizer.from_pretrained(f"{team}/{model_name}")


def get_auto_model(
    team: str,
    model_name: str,
    step: int | None,
    torch_dtype="auto",
    device: str = "cuda",
):
    kwargs = {"torch_dtype": torch_dtype}
    if step:
        kwargs["revision"] = f"step{step}"

    return AutoModelForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).to(
        device
    )


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
