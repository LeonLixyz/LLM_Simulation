import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
from tqdm import tqdm

# Set up cache directories
HF_HOME_DIR = "/shared/share_mala/Leon/huggingface/cache"
TRANSFORMERS_CACHE_DIR = "/shared/share_mala/Leon/huggingface/cache"
os.makedirs(HF_HOME_DIR, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = HF_HOME_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR


def load_llm(model_name, backend):

    if backend == "vllm":
        llm = LLM(model=model_name)
    elif backend == "openai":
        llm = OpenAI(model=model_name)
    elif backend == "anthropic":
        llm = Anthropic(model=model_name)
    elif backend == "groq":
        llm = Groq(model=model_name)
    elif backend == "mistral":
        llm = Mistral(model=model_name)
    elif backend == "openai":
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        return tokenizer
    else:
        raise ValueError(f"Invalid backend: {backend}")

    return llm
    