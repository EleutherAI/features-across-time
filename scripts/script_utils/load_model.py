import os
from pathlib import Path

import torch

# from mamba_model import MambaModel
# from mamba_ssm import MambaLMHeadModel
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


# def get_zyphra_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
#     kwargs = {"device": "cuda"}
#     if step:
#         kwargs["iteration"] = step

#     return MambaLMHeadModel.from_pretrained(f"{team}/{model_name}", **kwargs)


# def get_black_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
#     return MambaModel.from_pretrained(
#         pretrained_model_name=f"{team}/{model_name}"
#     ).cuda().half()


# def get_hails_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
#     kwargs = {"cache_dir": cache_dir}
#     if step:
#         kwargs["revision"] = f"step{step}"

#     return MambaForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).cuda()


def get_auto_model(
    team: str,
    model_name: str,
    step: int | None,
    cache_dir: str,
    torch_dtype="auto",
    device: str = "cuda",
):
    kwargs = {"torch_dtype": torch_dtype, "cache_dir": cache_dir}
    if step:
        kwargs["revision"] = f"step{step}"

    return AutoModelForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).to(
        device
    )


def load_and_combine_shards(model_dir, step, device):
    shards = [
        file
        for file in os.listdir(f"{model_dir}/checkpoint-{step}")
        if "pytorch_model-" in file
    ]
    state_dict = {}
    for shard in shards:
        shard_path = os.path.join(model_dir, f"checkpoint-{step}", shard)
        partial_state_dict = torch.load(shard_path, map_location=device)
        state_dict.update(partial_state_dict)

    return state_dict


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
