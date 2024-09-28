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
os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,6,7'


def load_llm(model_name):
    try:
        llm = LLM(model=model_name, tensor_parallel_size=4, gpu_memory_utilization=0.95, download_dir=HF_HOME_DIR)
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

def generate_text(llm, prompt, max_tokens=1000, temperature=0.95, top_p=0.95):
    sampling_params = SamplingParams(
        temperature=temperature,
        top_p=top_p,
        max_tokens=max_tokens
    )
    outputs = llm.generate([prompt], sampling_params)
    return outputs[0].outputs[0].text

def generate_topics(data):
    
    topic = data['topic']
    candidates = data['candidates']
    message = "### QUESTION ###\n\n"
    message += f"{topic}\n\n"
    
    if 'question' in data:
        question = data['question']
        message += f"Question: {question}\n\n"
    
    message += "### CHOICES ###:\n\n"
    for choice, info in candidates.items():
        stance = info['stance']
        message += f"- {choice}: {stance}\n\n"
        
    with open("/user/al4263/Simulate/Prompts/Pew_Research/opinion_simulation/user_instruction.json", "r") as f:
        instruction = json.load(f)
        message += instruction["Instruction"]

    return message

def load_persona(index):
    file_path = f"/user/al4263/Simulate/Pew_Research/ATP_W82/persona_llama/persona_{index}.json"
    with open(file_path, "r") as f:
        persona_data = json.load(f)
    return persona_data

def generate_opinion(llm, tokenizer, index, formatted_topic, system_message):
    persona_system = system_message["system_message"].format(PERSONA=load_persona(index))
    user_message = formatted_topic

    messages = [
        {"role": "system", "content": persona_system},
        {"role": "user", "content": user_message}
    ]
    prompt = tokenizer.apply_chat_template(messages, tokenize=False)
    output = generate_text(llm, prompt, max_tokens=1000)
    
    return output

def extract_reasoning_and_decision(output):
    reasoning_parts = output.lower().split("reasoning:")
    answer_parts = output.lower().split("answer:")
    
    if len(reasoning_parts) > 1:
        reasoning = reasoning_parts[1].split("\n")[0].strip()
    else:
        reasoning = ""
    
    if len(answer_parts) > 1:
        decision = answer_parts[1].strip()
    else:
        decision = output.strip()
    
    return reasoning, decision

def generate_opinions(llm, tokenizer, formatted_topic, system_message, topic_data, topic_persona_indices, num_personas):
    results = {
        "topic": topic_data["topic"],
        "opinions": [],
        "counts": {candidate['stance']: 0 for candidate in topic_data['candidates'].values()}
    }

    if num_personas == "max":
        indices_to_simulate = topic_persona_indices
    else:
        indices_to_simulate = topic_persona_indices[:min(num_personas, len(topic_persona_indices))]
    
    for index in tqdm(indices_to_simulate, desc=f"Generating opinions"):
        opinion = generate_opinion(llm, tokenizer, index, formatted_topic, system_message)
        reasoning, decision = extract_reasoning_and_decision(opinion)
        decision_letter = decision.strip()[0].upper()

        persona_id = f"{index:02d}"
        results["opinions"].append({
            "id": persona_id,
            "persona": load_persona(index),
            "reason": reasoning,
            "decision": decision_letter,
            "choice": topic_data['candidates'][decision_letter]['stance'] if decision_letter in topic_data['candidates'] else "Invalid/Abstain"
        })
        
        if decision_letter in topic_data['candidates']:
            stance = topic_data['candidates'][decision_letter]['stance']
            results["counts"][stance] += 1
        else:
            # Handle cases where the decision doesn't match any candidate
            results["counts"].setdefault("Invalid/Abstain", 0)
            results["counts"]["Invalid/Abstain"] += 1
    
    return results

def main():
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"

    llm = load_llm(model_name)
    if llm is None:
        print("Failed to load LLM. Exiting.")
        return

    tokenizer = load_tokenizer(model_name)
    if tokenizer is None:
        print("Failed to load tokenizer. Exiting.")
        return
    
    # Set up the parameters
    meta_stats = {}
    # num_personas = "max"
    num_personas = 100
    Topic = "ATP_W82"
    topic_start_index = 1


    with open(f"/user/al4263/Simulate/Simulations/Pew_Research/{Topic}/context.json", "r") as f:
        data = json.load(f)

    with open(f"/user/al4263/Simulate/Simulations/Pew_Research/{Topic}/non_nan_indices.json", "r") as f:
        persona_indices = json.load(f)

    with open("/user/al4263/Simulate/Prompts/Pew_Research/opinion_simulation/system_instruction.json", "r") as f:
        system_message = json.load(f)

    for topic_index in range(topic_start_index, len(data) + 1):
        formatted_topic = generate_topics(data[f"topic_{topic_index}"])
        topic_persona_indices = persona_indices[data[f"topic_{topic_index}"]["question_id"]]
        results = generate_opinions(llm, tokenizer, formatted_topic, system_message, data[f"topic_{topic_index}"], topic_persona_indices, num_personas)

        # Save results to a JSON file
        output_dir = f"/user/al4263/Simulate/Simulations/Pew_Research/{Topic}/Persona_Llama_Based"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/topic_{topic_index}_opinions.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        meta_stats[f"topic_{topic_index}"] = {
            "question_id": data[f"topic_{topic_index}"]["question_id"],
            "Topic": data[f"topic_{topic_index}"]["topic"],
            **{choice: count for choice, count in results['counts'].items()}
        }

        print(meta_stats)
    
    meta_stats_file = f"{output_dir}/meta_stats.json"
    with open(meta_stats_file, "w") as f:
        json.dump(meta_stats, f, indent=2)

    print(f"Meta stats saved to {meta_stats_file}")


if __name__ == "__main__":
    main()