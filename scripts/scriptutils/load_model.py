from mamba_model import MambaModel
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer, MambaForCausalLM, AutoModelForCausalLM, LlamaTokenizer


def get_neo_tokenizer():
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def get_amber_tokenizer():
    return LlamaTokenizer.from_pretrained("LLM360/Amber")


def get_zyphra_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
    kwargs = {"device": "cuda"}
    if step:
        kwargs["iteration"] = step

    return MambaLMHeadModel.from_pretrained(f"{team}/{model_name}", **kwargs)


def get_black_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
    return MambaModel.from_pretrained(
        pretrained_model_name=f"{team}/{model_name}"
    ).cuda().half()


def get_hails_mamba(team: str, model_name: str, step: int | None, cache_dir: str):
    kwargs = {"cache_dir": cache_dir}
    if step:
        kwargs["revision"] = f"step{step}"

    return MambaForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).cuda()


def get_auto_model(team: str, model_name: str, step: int | None, cache_dir: str):
    kwargs = {"torch_dtype": "auto", "cache_dir": cache_dir}
    if step:
        kwargs["revision"] = f"step{step}"
        
    return AutoModelForCausalLM.from_pretrained(f"{team}/{model_name}", **kwargs).cuda()