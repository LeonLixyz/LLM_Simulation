import os
import json
from transformers import AutoTokenizer
import us
from tqdm import tqdm
from anthropic import AnthropicBedrock
import argparse
from vllm import LLM, SamplingParams


# Set up cache directories
HF_HOME_DIR = "/shared/share_mala/Leon/huggingface/cache"
TRANSFORMERS_CACHE_DIR = "/shared/share_mala/Leon/huggingface/cache"
os.makedirs(HF_HOME_DIR, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = HF_HOME_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR

def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_llm(model_name, gpu_memory_utilization):
    print(f"Loading model {model_name} with GPU memory utilization {gpu_memory_utilization}")
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

def generate_topics(data):
    
    topic = data['topic']
    candidates = data['candidates']
    simulation_message = f"{topic}\n\n"
    
    if 'question' in data:
        question = data['question']
        simulation_message += f"Question: {question}\n\n"
    
    simulation_message += "### CHOICES ###:\n\n"
    for choice, info in candidates.items():
        stance = info['stance']
        simulation_message += f"- {choice}: {stance}\n\n"
    return simulation_message

def load_persona(Topic, persona_model, persona_type, index):
    file_path = f"/user/al4263/Simulate/Pew_Research/{Topic}/{persona_model}/{persona_type}/persona_{index}.json"
    with open(file_path, "r") as f:
        persona_data = json.load(f)
    return persona_data["PERSONA"]

def generate_opinion(llm, tokenizer, Topic, persona_model, persona_type, index, simulation_message, system_message, model_backend, temperature, top_p, max_tokens, fast_simulation):
    persona_system = system_message["system_message"]

    if fast_simulation:
        with open("/user/al4263/Simulate/Prompts/Pew_Research/opinion_simulation/fast_user_instruction.json", "r") as f:
            instruction = json.load(f)
    else:
        with open("/user/al4263/Simulate/Prompts/Pew_Research/opinion_simulation/reason_user_instruction.json", "r") as f:
            instruction = json.load(f)

    user_message = instruction["instructions"].format(QUESTION=simulation_message, PERSONA=load_persona(Topic, persona_model, persona_type, index))
    print(f'user_message: {user_message}')

    if model_backend == "vllm":
        messages = [
            {"role": "system", "content": persona_system},
            {"role": "user", "content": user_message}
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
                system = persona_system,
                messages=[{"role": "user", "content": user_message}],
                model="anthropic.claude-3-sonnet-20240229-v1:0",
            )
    
        return output.content[0].text

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

def generate_opinions(llm, tokenizer, Topic, persona_model, simulation_message, system_message, topic_data, topic_persona_indices, num_personas, persona_type, model_backend, temperature, top_p, max_tokens, fast_simulation):
    results = {
        "topic": topic_data["topic"],
        "opinions": [],
        "counts": {candidate['stance']: 0 for candidate in topic_data['candidates'].values()},
        "percentage": {}
    }

    if num_personas == -1:
        indices_to_simulate = topic_persona_indices
    else:
        indices_to_simulate = [idx for idx in topic_persona_indices if idx < num_personas]

    # for index in tqdm(indices_to_simulate, desc=f"Generating opinions"):

    for index in tqdm(range(num_personas), desc=f"Generating opinions"):
        opinion = generate_opinion(llm = llm, 
                                   tokenizer = tokenizer, 
                                   Topic=Topic,
                                   persona_model=persona_model,
                                   persona_type=persona_type,
                                   index=index,
                                   simulation_message=simulation_message,
                                   system_message=system_message,
                                   model_backend=model_backend,
                                   temperature=temperature,
                                   top_p=top_p,
                                   max_tokens=max_tokens,
                                   fast_simulation=fast_simulation)
        reasoning, decision = extract_reasoning_and_decision(opinion)
        decision_letter = decision.strip()[0].upper()

        persona_id = f"{index:02d}"
        results["opinions"].append({
            "id": persona_id,
            "persona": load_persona(Topic, persona_model, persona_type, index),
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
    
    total_responses = sum(results["counts"].values())
    results["percentage"] = {stance: round((count / total_responses * 100), 2) for stance, count in results["counts"].items()}

    # Sort counts and percentages in descending order
    results["counts"] = dict(sorted(results["counts"].items(), key=lambda x: x[1], reverse=True))
    results["percentage"] = dict(sorted(results["percentage"].items(), key=lambda x: x[1], reverse=True))

    return results

def main():

    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--model_backend", required=True, help="Model backend")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--model_save_name", required=True, help="Model save name")
    parser.add_argument("--persona_model", required=True, help="Persona model")
    parser.add_argument("--num_personas", type=int, required=True, help="Number of personas")
    parser.add_argument("--Topic", required=True, help="Topic")
    parser.add_argument("--persona_type", required=True, help="Persona type")
    parser.add_argument("--topic_start_index", type=int, required=True, help="Topic start index")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Top p")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max tokens")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, required=False, help="GPU memory utilization (default: 0.95)")    
    parser.add_argument("--fast_simulation", type=bool, default=False, required=True, help="Use fast user instruction (default: False)")
    args = parser.parse_args()

    # Use args directly instead of loading from config
    model_backend = args.model_backend
    model_name = args.model_name
    model_save_name = args.model_save_name
    persona_model = args.persona_model
    num_personas = args.num_personas
    Topic = args.Topic
    persona_type = args.persona_type
    topic_start_index = args.topic_start_index
    temperature = args.temperature
    top_p = args.top_p
    max_tokens = args.max_tokens
    gpu_memory_utilization = args.gpu_memory_utilization
    fast_simulation = args.fast_simulation
    
    if model_backend == "vllm":
        llm = load_llm(model_name, gpu_memory_utilization)
        if llm is None:
            print("Failed to load LLM. Exiting.")
            return
        tokenizer = load_tokenizer(model_name)
        if tokenizer is None:
            print("Failed to load tokenizer. Exiting.")
            return
    elif model_backend == "AWS":
        llm = AnthropicBedrock(
        aws_access_key="AKIA55F4LB62PM5RETUQ",
        aws_secret_key="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",
        aws_region="us-west-2",
    )
        tokenizer = None


    
    # Set up the parameters
    meta_stats = {}
    # num_personas = "max"


    with open(f"/user/al4263/Simulate/Simulations/Pew_Research/{Topic}/context.json", "r") as f:
        data = json.load(f)

    with open(f"/user/al4263/Simulate/Simulations/Pew_Research/{Topic}/non_nan_indices.json", "r") as f:
        persona_indices = json.load(f)

    with open("/user/al4263/Simulate/Prompts/Pew_Research/opinion_simulation/system_instruction.json", "r") as f:
        system_message = json.load(f)

    for topic_index in range(topic_start_index, len(data) + 1):
        simulation_message = generate_topics(data[f"topic_{topic_index}"])
        topic_persona_indices = persona_indices[data[f"topic_{topic_index}"]["question_id"]]
        results = generate_opinions(llm = llm,
                                    tokenizer=tokenizer,
                                    Topic=Topic,
                                    simulation_message=simulation_message,
                                    system_message=system_message,
                                    topic_data=data[f"topic_{topic_index}"],
                                    persona_model=persona_model,
                                    topic_persona_indices=topic_persona_indices,
                                    num_personas=num_personas,
                                    persona_type=persona_type,
                                    model_backend=model_backend,
                                    temperature=temperature,
                                    top_p=top_p,
                                    max_tokens=max_tokens,
                                    fast_simulation=fast_simulation)

        # Save results to a JSON file
        output_dir = f"/user/al4263/Simulate/Simulations/Pew_Research/{Topic}/{model_save_name}/{persona_model}_{persona_type}"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/topic_{topic_index}_opinions.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        meta_stats[f"topic_{topic_index}"] = {
            "question_id": data[f"topic_{topic_index}"]["question_id"],
            "Topic": data[f"topic_{topic_index}"]["topic"],
            "counts": {choice: count for choice, count in results['counts'].items()},
            "percentage": {choice: percentage for choice, percentage in results['percentage'].items()}
        }

        print(meta_stats)
    
    meta_stats_file = f"{output_dir}/meta_stats.json"
    with open(meta_stats_file, "w") as f:
        json.dump(meta_stats, f, indent=2)

    print(f"Meta stats saved to {meta_stats_file}")


if __name__ == "__main__":
    main()