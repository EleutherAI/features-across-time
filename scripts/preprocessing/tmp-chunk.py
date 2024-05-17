from datasets import load_dataset, DownloadConfig
import numpy as np
from transformers import AutoTokenizer
from chunk import chunk_and_tokenize
import os

output_dir = '/mnt/ssd-1/lucia'

# dataset = load_dataset(
#     "allenai/c4", 
#     "es", 
#     split='train', 
#     download_config=DownloadConfig(resume_download=True),
#     cache_dir=f"/mnt/ssd-1/hf_cache")

dataset = load_dataset(
    "spanish_billion_words", 
    split='train', 
    download_config=DownloadConfig(resume_download=True),
    cache_dir=f"/mnt/ssd-1/hf_cache")

model_name = "pythia-160m"
tokenizer = AutoTokenizer.from_pretrained(f"EleutherAI/{model_name}")   
# filtered_dataset = dataset.shuffle().select(range(int(3.5e11)))
shuffled = dataset.shuffle()
data, bpb_ratio = chunk_and_tokenize(shuffled, tokenizer, max_length=2049)

data.save_to_disk(f"{output_dir}/es_1b_full_tokenized.hf")

fp = np.memmap(f'{output_dir}/es_1b_full.bin', dtype=np.uint16, mode='w+', shape=(len(data), 2049))
for i, item in enumerate(data):
    fp[i] = item['input_ids'].numpy()   
fp.flush()
