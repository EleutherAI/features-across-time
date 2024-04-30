import os
from safetensors import safe_open
import torch

# from mamba_model import MambaModel
# from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer, GPTNeoXForCausalLM, AutoModel

from scripts.script_utils.custom_neo import GPTNeoXForCausalLMWithBias

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


def get_auto_model(team: str, model_name: str, step: int | None, cache_dir: str, torch_dtype="auto", device: str = "cuda"):
    kwargs = {"torch_dtype": torch_dtype, "cache_dir": cache_dir}
    if step:
        kwargs["revision"] = f"step{step}"

    return AutoModelForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).to(device)    


def get_gpt_neo_with_bias(team: str, model_name: str, step: int | None, cache_dir: str, device: str = "cuda"):
    kwargs = {"torch_dtype": "auto", "cache_dir": cache_dir, "ignore_mismatched_sizes": True}
    if step:
        kwargs["revision"] = f"step{step}"

    return GPTNeoXForCausalLMWithBias.from_pretrained(f"{team}/{model_name}", **kwargs).to(device)    

def load_and_combine_shards(model_dir, step, device):
    shards = [file for file in os.listdir(f"{model_dir}/checkpoint-{step}") if 'pytorch_model-' in file]
    state_dict = {}
    for shard in shards:
        shard_path = os.path.join(model_dir, f"checkpoint-{step}", shard)
        partial_state_dict = torch.load(shard_path, map_location=device)
        state_dict.update(partial_state_dict)
    
    return state_dict

def get_es_finetune(team: str, model_name: str, step: int | None, cache_dir: str, device: str = "cuda"):
    model_dir = f"/mnt/ssd-1/lucia/{model_name}-es"
    assert os.path.isdir(model_dir), f"Local directory {model_dir} does not exist"
    # return GPTNeoXForCausalLM.from_pretrained(
    #     f'{model_dir}/checkpoint-{step}', 
    #     torch_dtype="auto", cache_dir=cache_dir
    # ).to(device)

    try:
        model_path = f"{model_dir}/checkpoint-{step}/model.safetensors"
        new_state_dict = {}
        with safe_open(model_path, framework="pt", device="cpu") as f:
            for key in f.keys():
                new_state_dict[key.replace("_orig_mod.", "")] = f.get_tensor(key)
    except:
        try:
            model_path = f"{model_dir}/checkpoint-{step}/pytorch_model.bin"
            state_dict = torch.load(model_path)
            new_state_dict = {}
            for key in state_dict.keys():
                new_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]
        except:
            model_path = f"{model_dir}/checkpoint-{step}"
            state_dict = load_and_combine_shards(model_dir=model_dir, step=step, device=device)
            for key in state_dict.keys():
                new_state_dict[key.replace("_orig_mod.", "")] = state_dict[key]

    model = GPTNeoXForCausalLM.from_pretrained(
        f"{model_dir}/checkpoint-{step}",
        torch_dtype="auto",
        cache_dir=cache_dir
    ).to(device)    
    model.load_state_dict(new_state_dict)
    return model
