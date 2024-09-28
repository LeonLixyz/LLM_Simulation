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
        llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=0.9, download_dir=HF_HOME_DIR)
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

def generate_topics(topic_data):
    messages = []
    
    for subtopic, data in topic_data.items():
        if subtopic.startswith('topic_'):
            topic = data['topic']
            question = data['question']
            candidates = data['candidates']
            
            message = "###QUESTION###\n\n"
            message += f"Topic: {topic}\n\n"
            message += f"Question: {question}\n\n"
            
            message += "Candidates' Stances:\n"
            for choice, info in candidates.items():
                name = info['name']
                background = info['background_info']
                stance = info['stance']
                message += f"- {choice}: {name} ({background})\n  Stance: {stance}\n\n"
            
        with open("/user/al4263/Simulate/Persona/prompts/opinion_simulation/user_instruction.json", "r") as f:
            instruction = json.load(f)
            message += instruction["Instruction"]
            
            messages.append(message)
    
    return messages

def load_persona(state, index):
    file_path = f"/user/al4263/Simulate/Persona/Persona_GPT_Based/{state.abbr}/{state.abbr}_persona_{index}.json"
    with open(file_path, "r") as f:
        persona_data = json.load(f)
    return persona_data

def generate_opinion(llm, tokenizer, state, index, topic, system_message):
    persona_system = system_message["system_message"].format(PERSONA=load_persona(state, index)["PERSONA"])
    user_message = topic

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

def generate_opinions_for_state(llm, tokenizer, state, num_personas, topic, system_message):
    results = {
        "topic": topic,
        "opinions": [],
        "counts": {"A": 0, "B": 0, "ABSTAIN": 0}
    }
    
    for index in tqdm(range(num_personas), desc=f"Generating opinions for {state.name}"):
        opinion = generate_opinion(llm, tokenizer, state, index, topic, system_message)
        reasoning, decision = extract_reasoning_and_decision(opinion)
        
        persona_id = f"{state.abbr}_{index:02d}"
        results["opinions"].append({
            "id": persona_id,
            "persona": load_persona(state, index)["PERSONA"],
            "reason": reasoning,
            "decision": decision
        })
        
        # Count the decisions
        decision_letter = decision.strip()[0].upper()
        if decision_letter in ['A', 'B']:
            results["counts"][decision_letter] += 1
        else:
            results["counts"]["ABSTAIN"] += 1
    
    return results

def main():
    model_name = "meta-llama/Meta-Llama-3.1-70B-Instruct"
    llm = load_llm(model_name)
    tokenizer = load_tokenizer(model_name)

    if llm is None or tokenizer is None:
        print("Failed to load model or tokenizer. Exiting.")
        return
    
    # Set up the parameters
    meta_stats = {}
    state = us.states.TX
    num_personas = 200


    # load topic from prompts/election/economic_policy.json
    with open("/user/al4263/Simulate/Persona/prompts/election/economic_policy.json", "r") as f:
        data = json.load(f)

    with open("/user/al4263/Simulate/Persona/prompts/opinion_simulation/system_instruction.json", "r") as f:
        system_message = json.load(f)

    for topic_index in range(len(generate_topics(data))):
        topic = generate_topics(data)[topic_index]
        # Generate opinions
        results = generate_opinions_for_state(llm, tokenizer, state, num_personas, topic, system_message)

        # Save results to a JSON file
        output_dir = f"/user/al4263/Simulate/Simulations/Election/Economic_Policy/Persona_GPT_Based"
        os.makedirs(output_dir, exist_ok=True)
        output_file = f"{output_dir}/topic_{topic_index}_opinions_{state.abbr}.json"
        with open(output_file, "w") as f:
            json.dump(results, f, indent=2)

        meta_stats[f"topic_{topic_index}"] = {
            "Topic": topic,
            "Trump (A)": results['counts']['A'],
            "Harris (B)": results['counts']['B'],
            "Abstain": results['counts'].get('ABSTAIN', 0)
        }

        print(meta_stats)
    
    meta_stats_file = f"{output_dir}/meta_stats_{state.abbr}.json"
    with open(meta_stats_file, "w") as f:
        json.dump(meta_stats, f, indent=2)

    print(f"Meta stats saved to {meta_stats_file}")


if __name__ == "__main__":
    main()