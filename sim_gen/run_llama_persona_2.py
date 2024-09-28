import json
import os
CONFIG_PATH = "configs/llama_70B_persona_config.json"
TOPICS = ["ATP_W82"]
PERSONA_TYPES = ["vivid_persona", "factual_persona", "table_persona", "vanilla_persona"]
# PERSONA_TYPES = ["vivid_persona"]
def load_config():
    with open(CONFIG_PATH, 'r') as f:
        return json.load(f)

def create_command(base_config, topic, persona_type):
    command = "python generate_persona.py"
    args = base_config.copy()
    args["Topic"] = topic
    args["persona_type"] = persona_type
    args["num_personas"] = 200
    args["start_index"] = 100
    args["gpu_memory_utilization"] = 0.7
    
    for key, value in args.items():
        command += f" --{key} {value}"
    
    return command



def main():
    base_config = load_config()

    for topic in TOPICS:
        for persona_type in PERSONA_TYPES:
            command = create_command(base_config, topic, persona_type)
            print(f"Running: {command}")
            os.system(command)

if __name__ == "__main__":
    main()