import torch
import torch.nn as nn
from transformers import GPTNeoXForCausalLM
from transformers.activations import ACT2FN

class GPTNeoXForCausalLMWithBias(GPTNeoXForCausalLM):
    def __init__(self, config):
        super().__init__(config)  
        self.embed_out.bias = nn.Parameter(torch.empty(self.embed_out.out_features))

    def forward(self, input_ids, **kwargs):
        return super().forward(input_ids, **kwargs)
