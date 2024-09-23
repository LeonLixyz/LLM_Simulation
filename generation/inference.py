import os
import argparse
from datasets import load_dataset, load_from_disk
from vllm import LLM, SamplingParams
from tqdm import tqdm

HF_HOME_DIR = "/shared/share_mala/Leon/huggingface/cache"
TRANSFORMERS_CACHE_DIR = "/shared/share_mala/Leon/huggingface/cache"
if not os.path.exists(HF_HOME_DIR):
    os.makedirs(HF_HOME_DIR)
if not os.path.exists(TRANSFORMERS_CACHE_DIR):
    os.makedirs(TRANSFORMERS_CACHE_DIR)

os.environ['HF_HOME'] = HF_HOME_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR   

def calculate_reward(text, llm, sampling_params):
    outputs = llm.generate([text], sampling_params)
    logprobs = outputs[0].prompt_logprobs
    return sum(next(iter(prob.values())).logprob for prob in logprobs[1:])

def load_llm(model_name):
    try:
        llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=0.9, download_dir=HF_HOME_DIR)
    except Exception as e:
        print(f"Error initializing LLM: {e}")
    return llm

def main(use_rlhf, continue_from_scratch):
    load_dir = "llama3.1_DPO_reward_dataset"
    save_dir = "llama3.1_DPO_reward_dataset_new"
    
    if continue_from_scratch:
        print("Loading dataset from disk...")
        ds = load_from_disk(load_dir)
    else:
        print("Loading dataset from Hugging Face...")
        ds = load_dataset("andrewsiah/embed-exploration-bench")
    
    if use_rlhf:
        model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
        column_prefix = "llama3.1_70B_instruct_logprob"
        print("Using RLHF model: ", model_name)
    else:
        model_name = "meta-llama/Meta-Llama-3.1-70B"
        column_prefix = "llama3.1_70B_logprob"
        print("Using pretrained model: ", model_name)
    
    llm = load_llm(model_name)
    sampling_params = SamplingParams(temperature=0.0, top_p=1.0, prompt_logprobs=0, max_tokens=1)

    # Determine the starting point for column addition
    existing_columns = [col for col in ds['train'].column_names if col.startswith(column_prefix)]
    start_index = len(existing_columns) + 1

    for i in tqdm(range(start_index, 101), desc="Calculating rewards"):
        column_name = f"prompt_response_{i}"
        reward_column = f"{column_prefix}_{i}"
        print(f"Calculating rewards for column: {column_name}")
        rewards = [calculate_reward(text, llm, sampling_params) for text in ds['train'][column_name]]
        ds['train'] = ds['train'].add_column(reward_column, rewards)
        
        # Save after each new column is added
        ds.save_to_disk(save_dir)
        print(f"Dataset saved to: {os.path.abspath(save_dir)}")

    print("All rewards calculated and dataset saved.")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Calculate rewards using LLaMA model")
    parser.add_argument('--use_rlhf', action='store_true', help='Use RLHF\'d (instruct) model instead of pretrained')
    parser.add_argument('--continue', dest='continue_from_scratch', action='store_true', help='Continue from existing dataset')
    args = parser.parse_args()

    main(args.use_rlhf, args.continue_from_scratch)