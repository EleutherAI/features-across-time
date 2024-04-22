# from mamba_model import MambaModel
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer, AutoModelForCausalLM, LlamaTokenizer # MambaForCausalLM

from script_utils.custom_neo import GPTNeoXForCausalLMWithBias

def get_auto_tokenizer(team: str, model_name: str):
    return AutoTokenizer.from_pretrained(f"{team}/{model_name}")


def get_amber_tokenizer(team: str, model_name: str):
    return LlamaTokenizer.from_pretrained(f"{team}/{model_name}")


def get_zyphra_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
    kwargs = {"device": "cuda"}
    if step:
        kwargs["iteration"] = step

    return MambaLMHeadModel.from_pretrained(f"{team}/{model_name}", **kwargs)


# def get_black_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
#     return MambaModel.from_pretrained(
#         pretrained_model_name=f"{team}/{model_name}"
#     ).cuda().half()


# def get_hails_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
#     kwargs = {"cache_dir": cache_dir}
#     if step:
#         kwargs["revision"] = f"step{step}"

#     return MambaForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).cuda()


def get_auto_model(team: str, model_name: str, step: int | None, cache_dir: str, torch_dtype="auto"):
    kwargs = {"torch_dtype": torch_dtype, "cache_dir": cache_dir}
    if step:
        kwargs["revision"] = f"step{step}"
        
    return AutoModelForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).cuda()    


def get_gpt_neo_with_bias(team: str, model_name: str, step: int | None, cache_dir: str):
    kwargs = {"torch_dtype": "auto", "cache_dir": cache_dir, "ignore_mismatched_sizes": True}
    if step:
        kwargs["revision"] = f"step{step}"

    return GPTNeoXForCausalLMWithBias.from_pretrained(f"{team}/{model_name}", **kwargs).cuda()    


def get_es_finetune(team: str, model_name: str, step: int | None, cache_dir: str):
    assert model_name == "pythia-160m" and team == "EleutherAI", "Spanish finetune of requested model not available locally"

    kwargs = {"torch_dtype": "auto", "cache_dir": cache_dir}
    if step:
        kwargs["revision"] = f"step{step}"
        
    return AutoModelForCausalLM.from_pretrained(f"/mnt/ssd-1/lucia/es-160m/checkpoint-{step}", **kwargs).cuda()    
