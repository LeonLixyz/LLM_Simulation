import json
import os
import us

HF_HOME_DIR = "./huggingface/cache"
os.environ['HF_HOME'] = HF_HOME_DIR

BASE_CONFIG_PATH = "configs/persona_config.json"
LLM_CONFIGS_PATH = "model_configs"
persona_type_list = ["objective_table_persona", "subjective_table_persona", "descriptive_persona"]

LLM_MODELS = [
    "Llama-3.1-8B-Instruct",
    "Llama-3.1-70B-Instruct",
    "Athene-70B",
    "Mixtral-8x7B-Instruct-v0.1",
    "Nemotron-70B-Instruct",
    "Qwen2.5-72B-Instruct",
]

def load_config(path):
    with open(path, 'r') as f:
        return json.load(f)

def create_command(base_config, persona_type_list, llm_config):
    command = "python Persona_Generation.py"
    args = base_config.copy()
    args["model_backend"] = llm_config["model_backend"]
    args["model_name"] = llm_config["model_name"]
    args["model_save_name"] = llm_config["model_save_name"]
    args["persona_type_list"] = f'"{str(persona_type_list).replace(" ", "")}"'
    args["HF_HOME_DIR"] = HF_HOME_DIR
    for key, value in args.items():
        command += f" --{key} {value}"
    return command



def main():
    base_config = load_config(BASE_CONFIG_PATH)

    for llm_model in LLM_MODELS:    
        llm_config = load_config(f"{LLM_CONFIGS_PATH}/{llm_model}.json")
        command = create_command(base_config, persona_type_list, llm_config)
        print(f"Running: {command}")
        os.system(command)

if __name__ == "__main__":
    main()