import os
import json
from transformers import AutoTokenizer
import us
from anthropic import AnthropicBedrock
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams

# Set up cache directories
HF_HOME_DIR = "/shared/share_mala/Leon/huggingface/cache"
TRANSFORMERS_CACHE_DIR = "/shared/share_mala/Leon/huggingface/cache"
os.makedirs(HF_HOME_DIR, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = HF_HOME_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,5,6,7'

def load_llm(model_name, gpu_memory_utilization):
    try:
        llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=gpu_memory_utilization, download_dir=HF_HOME_DIR)
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

def generate_persona(llm, tokenizer, metadata, system_message, instructions, model_backend, temperature, top_p, max_tokens, template):

    if template is not None:
        user_prompt = instructions.format(METADATA=json.dumps(metadata, indent=2), TEMPLATE=json.dumps(template, indent=2))
    else:
        user_prompt = instructions.format(METADATA=json.dumps(metadata, indent=2))

    if model_backend == "vllm":
        messages = [
            {"role": "system", "content": system_message},
            {"role": "user", "content": user_prompt}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens
        )

        outputs = llm.generate([prompt], sampling_params)
        return outputs[0].outputs[0].text

    elif model_backend == "AWS":
        output = llm.messages.create(
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=top_p,
            system=system_message,
            messages=[{"role": "user", "content": user_prompt}],
            model="anthropic.claude-3-sonnet-20240229-v1:0",
        )
        return output.content[0].text

def extract_persona(output):
    persona_parts = output.lower().split("persona:")
    if len(persona_parts) > 1:
        return persona_parts[1].strip()
    else:
        output = output.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
        return output.strip()

def save_persona(index, persona, Topic, model_save_name, persona_type):
    base_dir = f"/user/al4263/Simulate/Pew_Research/{Topic}/{model_save_name}/{persona_type}"
    os.makedirs(base_dir, exist_ok=True)
    
    file_name = f"persona_{index}.json"
    file_path = os.path.join(base_dir, file_name)

    data = {
        "PERSONA": persona
    }
    
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)

def generate_personas(llm, tokenizer, Topic, num_personas, start_index, model_backend, temperature, top_p, max_tokens, model_save_name, persona_type):
    base_dir = f"/user/al4263/Simulate/Pew_Research/{Topic}/persona_meta"
    
    with open(f"/user/al4263/Simulate/Prompts/Pew_Research/persona_generation/{persona_type}/system.json", "r") as f:
        system_message = json.load(f)["system_message"]
    
    with open(f"/user/al4263/Simulate/Prompts/Pew_Research/persona_generation/{persona_type}/instructions.json", "r") as f:
        instructions = json.load(f)["instructions"]

    if persona_type == "table_persona":
        with open(f"/user/al4263/Simulate/Prompts/Pew_Research/persona_generation/{persona_type}/template.json", "r") as f:
            template = json.load(f)
        
    else:
        template = None

    if num_personas == -1:
        num_personas = len([f for f in os.listdir(base_dir) if f.endswith('.json')])

    for i in tqdm(range(start_index, num_personas), desc=f"Generating {Topic} personas"):
        file_name = f"persona_{i}.json"
        file_path = os.path.join(base_dir, file_name)
        
        with open(file_path, "r") as f:
            metadata = json.load(f)
        
        output = generate_persona(llm, tokenizer, metadata, system_message, instructions, model_backend, temperature, top_p, max_tokens, template)
        persona = extract_persona(output)    
        save_persona(i, persona, Topic, model_save_name, persona_type)

def main():
    parser = argparse.ArgumentParser(description="Persona Generation")
    parser.add_argument("--model_backend", required=True, help="Model backend")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--model_save_name", required=True, help="Model save name")
    parser.add_argument("--persona_type", required=True, help="Persona type")
    parser.add_argument("--num_personas", type=int, required=True, help="Number of personas")
    parser.add_argument("--Topic", required=True, help="Topic name")
    parser.add_argument("--start_index", type=int, required=True, help="Start index")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Top p")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max tokens")
    parser.add_argument("--gpu_memory_utilization", type=float, required=False, default=0.95, help="GPU memory utilization")
    
    args = parser.parse_args()

    if args.model_backend == "vllm":
        llm = load_llm(args.model_name, args.gpu_memory_utilization)
        tokenizer = load_tokenizer(args.model_name)
    elif args.model_backend == "AWS":
        llm = AnthropicBedrock(
            aws_access_key="AKIA55F4LB62PM5RETUQ",
            aws_secret_key="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",
            aws_region="us-west-2",
        )
        tokenizer = None

    generate_personas(llm, tokenizer, args.Topic, args.num_personas, args.start_index, 
                      args.model_backend, args.temperature, args.top_p, args.max_tokens, args.model_save_name, args.persona_type)

    print("Persona generation complete.")

if __name__ == "__main__":
    main()