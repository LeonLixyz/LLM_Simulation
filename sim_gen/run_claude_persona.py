import json
import subprocess
import concurrent.futures

BASE_CONFIG_PATH = "configs/claude_persona_config.json"
# PERSONA_TYPES = ["vivid_persona", "factual_persona"]
PERSONA_TYPES = ["table_persona", "vanilla_persona"]
TOPICS = ["ATP_W82"]
# PERSONA_TYPES = ["persona_llama"]

def load_base_config():
    with open(BASE_CONFIG_PATH, 'r') as f:
        return json.load(f)

def create_command(base_config, topic, persona_type):
    config = base_config.copy()
    config["Topic"] = topic
    config["persona_type"] = persona_type
    
    command = "python generate_persona.py"
    for key, value in config.items():
        command += f" --{key} {value}"
    
    return command

def run_inference(command):
    print(f"Running: {command}")
    try:
        subprocess.run(command, shell=True, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error running inference: {e}")

def main():
    base_config = load_base_config()
    commands = [create_command(base_config, topic, persona_type) 
                for topic in TOPICS 
                for persona_type in PERSONA_TYPES]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_inference, commands)

if __name__ == "__main__":
    main()