import os
import json
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer
import us
from tqdm import tqdm

# Set up cache directories
HF_HOME_DIR = "/shared/share_mala/Leon/huggingface/cache"
TRANSFORMERS_CACHE_DIR = "/shared/share_mala/Leon/huggingface/cache"
os.makedirs(HF_HOME_DIR, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = HF_HOME_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR

def load_llm(model_name):
    try:
        llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=0.95, download_dir=HF_HOME_DIR)
        print(f"Model {model_name} loaded successfully.")
        return llm
    except Exception as e:
        print(f"Error initializing LLM: {e}")
        return None

def load_tokenizer(model_name):
    try:
        tokenizer = AutoTokenizer.from_pretrained(model_name, use_fast=False)
        print(f"Tokenizer for {model_name} loaded successfully.")
        return tokenizer
    except Exception as e:
        print(f"Error loading tokenizer: {e}")
        return None

def generate_text(llm, prompt, max_tokens=1000, temperature=1.0, top_p=1.0):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

def extract_persona(output):
    # Split by "persona:" (case-insensitive), ignoring potential spaces
    persona_parts = output.lower().split("persona:")
    if len(persona_parts) > 1:
        return persona_parts[1].strip()
    else:
        # If no split occurred, return the entire output
        return output.strip()

def save_persona(state, index, persona):
    base_dir = f"/user/al4263/Simulate/Persona/Persona_Llama_Based/{state.abbr}"
    os.makedirs(base_dir, exist_ok=True)
    
    file_name = f"{state.abbr}_persona_{index}.json"
    file_path = os.path.join(base_dir, file_name)

    data = {
        "PERSONA": persona
    }
    
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)

def process_state_personas(llm, tokenizer, state, num_personas, start_index):
    base_dir = f"/user/al4263/Simulate/Persona/Persona_Meta/{state.abbr}"
    
    with open("/user/al4263/Simulate/Persona/prompts/persona_generation/persona_generation.json", "r") as f:
        system_message = json.load(f)["system_message"]
    
    for i in tqdm(range(start_index, num_personas), desc=f"Processing {state.name} personas"):
        file_name = f"{state.abbr}_persona_{i}.json"
        file_path = os.path.join(base_dir, file_name)
        
        with open(file_path, "r") as f:
            metadata = json.load(f)
        
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": f"### PERSONA METADATA ###\n\n{json.dumps(metadata, indent=2)}\n\n### PERSONA GENERATION        ###\n\n"}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        output = generate_text(llm, prompt)
        persona = extract_persona(output)    
        
        save_persona(state, i, persona)

def main():
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    llm = load_llm(model_name)
    tokenizer = load_tokenizer(model_name)

    if llm is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return

    num_personas = 1000  # Adjust this number as needed
    start_index = 0

    # for state in tqdm(us.states.STATES, desc="Processing states"):
    state = us.states.WY
    process_state_personas(llm, tokenizer, state, num_personas, start_index)

    print("Persona generation complete for all states.")

if __name__ == "__main__":
    main()