import json
import os

CONFIG_PATH = "configs/llama_70B_sim_config.json"
TOPICS = ["ATP_W82"]
PERSONA_MODELS = ["llama-70B-Instruct"]
PERSONA_TYPES = ["vivid_persona", "factual_persona", "vanilla_persona", "table_persona", "meta_persona"]
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def create_command(base_config, topic, persona_model, persona_type):
    command = "python sim.py"
    args = base_config.copy()
    args["Topic"] = topic
    args["persona_type"] = persona_type
    args["persona_model"] = persona_model
    
    for key, value in args.items():
        command += f" --{key} {value}"
    
    return command

def main():
    base_config = load_config()

    for topic in TOPICS:
        for persona_type in PERSONA_TYPES:
            for persona_model in PERSONA_MODELS:
                command = create_command(base_config, topic, persona_model, persona_type)
                print(command)
                os.system(command)


if __name__ == "__main__":
    main()