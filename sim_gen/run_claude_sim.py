import json
import subprocess
import concurrent.futures

BASE_CONFIG_PATH = "configs/claude_sim_config.json"
TOPICS = ["ATP_W82"]
PERSONA_MODELS = ["Claude"]
PERSONA_TYPES = ["vivid_persona", "factual_persona"]

def load_base_config():
    with open(BASE_CONFIG_PATH, 'r') as f:
        return json.load(f)

def create_command(base_config, topic, persona_model, persona_type):
    config = base_config.copy()
    config["Topic"] = topic
    config["persona_type"] = persona_type
    config["persona_model"] = persona_model
    command = "python sim.py"
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
    commands = [create_command(base_config, topic, persona_model, persona_type) 
                for topic in TOPICS 
                for persona_model in PERSONA_MODELS
                for persona_type in PERSONA_TYPES]

    with concurrent.futures.ProcessPoolExecutor() as executor:
        executor.map(run_inference, commands)

if __name__ == "__main__":
    main()