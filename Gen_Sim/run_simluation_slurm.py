import json
import os

HF_HOME_DIR = "./huggingface/cache"
os.environ['HF_HOME'] = HF_HOME_DIR
BASE_CONFIG_PATH = "configs/simulation_config.json"
LLM_CONFIGS_PATH = "model_configs"
persona_type_list = ["objective_table_persona", "subjective_table_persona", "descriptive_persona"]
persona_model_name_list = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Athene-70B",
    "Mixtral-8x7B-Instruct-v0.1",
    "Nemotron-70B-Instruct",
    "Qwen2.5-72B-Instruct",
]

LLM_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Athene-70B",
    "Mixtral-8x7B-Instruct-v0.1",
    "Nemotron-70B-Instruct",
    "Qwen2.5-72B-Instruct",
]

topic_list = ["Election"]

SLURM_TEMPLATE = """#!/bin/bash
#SBATCH -p columbia
#SBATCH -N 1
#SBATCH -c 96
#SBATCH --mem=2000000
#SBATCH --gpus=8
#SBATCH --job-name={job_name}
#SBATCH --output={log_dir}/{job_name}.out

source ~/.bashrc
conda activate vllm_env
{command}
"""

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_command(base_config, topic_list, persona_type_list, persona_model_name_list, llm_config):
    command = "python Opinion_Simulation.py"
    args = base_config.copy()
    args["topic_list"] = f'"{str(topic_list).replace(" ", "")}"'
    args["persona_type_list"] = f'"{str(persona_type_list).replace(" ", "")}"'
    args["persona_model_name_list"] = f'"{str(persona_model_name_list).replace(" ", "")}"'
    args["model_backend"] = llm_config["model_backend"]
    args["model_name"] = llm_config["model_name"]
    args["simulation_model_name"] = llm_config["model_save_name"]
    args["HF_HOME_DIR"] = HF_HOME_DIR
    for key, value in args.items():
        command += f" --{key} {value}"
    
    return command

def create_slurm_script(command, job_name, log_dir):
    return SLURM_TEMPLATE.format(job_name=job_name, log_dir=log_dir, command=command)

def main():
    log_dir = "./LLM_Simulation/slurm_run_logs"
    os.makedirs(log_dir, exist_ok=True)  # Create the directory if it doesn't exist

    base_config = load_config(BASE_CONFIG_PATH)
    for llm_model in LLM_MODELS:
        llm_config = load_config(f"{LLM_CONFIGS_PATH}/{llm_model}.json")
        command = create_command(base_config, topic_list, persona_type_list, persona_model_name_list, llm_config)
        job_name = f"sim_{llm_model}"
        slurm_script = create_slurm_script(command, job_name, log_dir)
        
        script_filename = f"slurm_{llm_model}.sh"
        with open(script_filename, 'w') as script_file:
            script_file.write(slurm_script)
        
        os.system(f"sbatch {script_filename}")

if __name__ == "__main__":
    main()