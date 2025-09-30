import os
import json
from transformers import AutoTokenizer
import us
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams
import ast

def load_llm(model_name, gpu_memory_utilization, HF_HOME_DIR):
    try:
        print(f"Using HF token: {os.environ.get('HUGGING_FACE_HUB_TOKEN', 'Not set')[:8]}...")
        
        llm = LLM(
            model=model_name, 
            tensor_parallel_size=8, 
            gpu_memory_utilization=gpu_memory_utilization, 
            download_dir=HF_HOME_DIR, 
            enable_prefix_caching=True,
            trust_remote_code=True
        )
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
    
def load_meta_persona(meta_data):
    persona_info =  meta_data.get("persona") or meta_data.get("PERSONA") or meta_data
    return persona_info

def generate_personas_batch(llm, tokenizer, num_personas, temperature, top_p, max_tokens, persona_type, state, meta_persona_dir, persona_prompt_dir, save_dir, model_save_name):
    meta_persona_path = meta_persona_dir.format(state=state)
    prompt_path = persona_prompt_dir.format(persona_type=persona_type)
    
    with open(f"{prompt_path}/system.json", "r") as f:
        system_message = json.load(f)["system_message"]
    
    with open(f"{prompt_path}/instructions.json", "r") as f:
        instructions = json.load(f)["instructions"]

    template = None
    if "table_persona" in persona_type:
        with open(f"{prompt_path}/template.json", "r") as f:
            template = json.load(f)

    if num_personas == -1:
        num_personas = len([f for f in os.listdir(meta_persona_path) if f.endswith('.json')])

    batch_prompts = []
    batch_indices = []
    sampling_params_list = []

    for i in range(num_personas):
        file_name = f"persona_{i}.json"
        file_path = os.path.join(meta_persona_path, file_name)
        
        with open(file_path, "r") as f:
            metadata = json.load(f)

        meta_persona = load_meta_persona(metadata)
        
        if template is not None:
            user_prompt = instructions.format(METADATA=json.dumps(meta_persona, indent=2), TEMPLATE=json.dumps(template, indent=2))
        else:
            user_prompt = instructions.format(METADATA=json.dumps(meta_persona, indent=2))

        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        
        prompt = tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        batch_prompts.append(prompt)
        batch_indices.append(i)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=i
        )
        sampling_params_list.append(sampling_params)

    outputs = llm.generate(batch_prompts, sampling_params_list)

    for output, index in zip(outputs, batch_indices):
        persona = extract_persona(output.outputs[0].text)
        save_persona(index, persona, persona_type, state, save_dir, model_save_name)

def extract_persona(output):
    persona_parts = output.split("Persona:")
    if len(persona_parts) > 1:
        return persona_parts[1].strip()
    else:
        return output.strip()

def save_persona(index, persona, persona_type, state, save_dir, model_save_name):
    base_dir = save_dir.format(state=state, persona_type=persona_type, model_save_name=model_save_name)
    os.makedirs(base_dir, exist_ok=True)
    
    file_name = f"persona_{index}.json"
    file_path = os.path.join(base_dir, file_name)

    data = {"PERSONA": persona}
    
    # Save the persona data
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)


def main():
    parser = argparse.ArgumentParser(description="Persona Generation")
    parser.add_argument("--model_backend", required=True, help="Model backend")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--model_save_name", required=True, help="Model save name")
    parser.add_argument("--persona_type_list", type=lambda x: ast.literal_eval(x.strip('"')),
                       help='List of persona types')
    parser.add_argument("--num_personas", type=int, required=True, help="Number of personas")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Top p")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max tokens")
    parser.add_argument("--HF_HOME_DIR", required=True, help="HF home directory")
    parser.add_argument("--meta_persona_dir", required=True, help="Meta persona directory")
    parser.add_argument("--persona_prompt_dir", required=True, help="Persona prompt directory")
    parser.add_argument("--save_dir", required=True, help="Save directory")
    parser.add_argument("--gpu_memory_utilization", type=float, required=False, default=0.95, help="GPU memory utilization")
    args = parser.parse_args()

    if args.model_backend == "vllm":
        print(f'loading {args.model_name}')
        llm = load_llm(args.model_name, args.gpu_memory_utilization, args.HF_HOME_DIR)
    tokenizer = load_tokenizer(args.model_name)

    for persona_type in args.persona_type_list:
        print(f'generating for {persona_type}')
        STATES = [state.abbr for state in us.states.STATES if state.abbr not in ['HI', 'AK']]
        for state in STATES:
            print(f'generating for {state}')
            generate_personas_batch(llm = llm, tokenizer = tokenizer, num_personas = args.num_personas, 
                                    temperature = args.temperature, top_p = args.top_p, max_tokens = args.max_tokens, 
                                    persona_type = persona_type, state = state, meta_persona_dir = args.meta_persona_dir, persona_prompt_dir = args.persona_prompt_dir, save_dir = args.save_dir, model_save_name = args.model_save_name)

        print("Persona generation complete.")

if __name__ == "__main__":
    main()