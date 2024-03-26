from mamba_model import MambaModel
from mamba_ssm import MambaLMHeadModel
from transformers import AutoTokenizer, MambaForCausalLM


def get_neo_tokenizer():
    return AutoTokenizer.from_pretrained("EleutherAI/gpt-neox-20b")


def get_zyphra_mamba(team: str, model_name: str, step: int):
    return MambaLMHeadModel.from_pretrained(
        f"{team}/{model_name}",
        iteration=step,
        device="cuda"
    )


def get_black_mamba(team: str, model_name: str, step: int):
    return MambaModel.from_pretrained(
        pretrained_model_name=f"{team}/{model_name}"
    ).cuda().half()


def get_hails_mamba(team: str, model_name: str, step: int):
    return MambaForCausalLM.from_pretrained(
        f"{team}/{model_name}", 
        revision=f"step{step}",
    ).cuda()