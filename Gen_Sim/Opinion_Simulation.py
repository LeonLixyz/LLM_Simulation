import os
import json
from transformers import AutoTokenizer
import us
import argparse
from vllm import LLM, SamplingParams
import glob
import ast
# set cuda visible devices
# os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1, 2, 3"


def load_config(config_path):
    with open(config_path, 'r') as f:
        return json.load(f)

def load_llm(model_name, gpu_memory_utilization, HF_HOME_DIR):
    print(f"Loading model {model_name} with GPU memory utilization {gpu_memory_utilization}")
    try:
        llm = LLM(model=model_name, tensor_parallel_size=8, gpu_memory_utilization=gpu_memory_utilization, download_dir=HF_HOME_DIR, enable_prefix_caching=True)
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

def load_persona(persona_dir, index):
    file_path = os.path.join(persona_dir, f"persona_{index}.json")
    with open(file_path, "r") as f:
        persona_data = json.load(f)
        
    persona_info =  persona_data.get("persona") or persona_data.get("PERSONA") or persona_data
    
    return persona_info

def load_persona(persona_dir, index):
    try:
        with open(f"{persona_dir}/persona_{index}.json", "r") as f:
            persona_data = json.load(f)
            persona_info =  persona_data.get("persona") or persona_data.get("PERSONA") or persona_data
        return persona_info
    except json.JSONDecodeError as e:
        print(f"Error loading persona {index}: {e}")
        return None  # or a default persona
    except FileNotFoundError:
        print(f"Persona directory: {persona_dir}")
        print(f"Persona file for index {index} not found")
        return None  # or a default persona

def generate_opinions_batch(llm, tokenizer, simulation_message, system_message, topic_data, persona_type, persona_dir, state, temperature, top_p, max_tokens, instruction):
    results = {
        "topic": topic_data["topic"],
        "opinions": [],
        "counts": {candidate['stance']: 0 for candidate in topic_data['candidates'].values()},
        "percentage": {}
    }

    persona_system = system_message["system_message"]

    batch_prompts = []
    batch_indices = []
    sampling_params_list = []


    # available_indices = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in glob.glob(f"{persona_dir}/persona_*.json")]
    # available_indices.sort()
    available_indices = list(range(1000))
    print(f'Simulating {len(available_indices)} personas for {state} - {persona_type}')

    for index in available_indices:
        user_message = instruction["instructions"].format(QUESTION=simulation_message, PERSONA=load_persona(persona_dir, index))
        messages = [
            {"role": "system", "content": persona_system},
            {"role": "user", "content": user_message}
        ]
        prompt = tokenizer.apply_chat_template(messages, tokenize=False)
        batch_prompts.append(prompt)
        batch_indices.append(index)

        sampling_params = SamplingParams(
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
            seed=index  
        )
        sampling_params_list.append(sampling_params)

    outputs = llm.generate(batch_prompts, sampling_params_list)

    for output, index in zip(outputs, batch_indices):
        opinion = output.outputs[0].text
        reasoning, decision_letter = extract_reasoning_and_decision(opinion)

        persona_id = f"{index:02d}"
        choice = topic_data['candidates'][decision_letter]['stance'] if decision_letter in topic_data['candidates'] else "Invalid/Abstain"
        if choice == "Invalid/Abstain":
            print(f"Invalid opinion for persona {persona_id}: {opinion}")
        
        results["opinions"].append({
            "id": persona_id,
            "persona": load_persona(persona_dir, index),
            "reason": reasoning,
            "decision": decision_letter,
            "choice": choice
        })
        if decision_letter in topic_data['candidates']:
            stance = topic_data['candidates'][decision_letter]['stance']
            results["counts"][stance] += 1
        else:
            results["counts"].setdefault("Invalid/Abstain", 0)
            results["counts"]["Invalid/Abstain"] += 1

    total_responses = sum(results["counts"].values())
    results["percentage"] = {stance: round((count / total_responses * 100), 2) for stance, count in results["counts"].items()}

    results["counts"] = dict(sorted(results["counts"].items(), key=lambda x: x[1], reverse=True))
    results["percentage"] = dict(sorted(results["percentage"].items(), key=lambda x: x[1], reverse=True))

    return results

def extract_reasoning_and_decision(output):
    # Strip the header if present
    output = output.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
    
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
    
    # Extract the first letter of the decision, if available
    decision_letter = decision[0].upper() if decision else ""
    
    return reasoning, decision_letter

def setup_model(args):
    """Initialize the LLM and tokenizer"""
    if args.model_backend == "vllm":
        llm = load_llm(args.model_name, args.gpu_memory_utilization, args.HF_HOME_DIR)
        tokenizer = load_tokenizer(args.model_name)
        if llm is None or tokenizer is None:
            print("Failed to load model components. Exiting.")
            return None, None
        return llm, tokenizer
    return None, None

def load_topic_resources(topic, args):
    """Load topic-specific data and instructions"""
    Topic_dir = args.topic_dir.format(Topic=topic)
    with open(f"{Topic_dir}/context.json", "r") as f:
        data = json.load(f)
    
    with open(f"{args.simulation_prompt_dir}/system_instruction.json", "r") as f:
        system_message = json.load(f)
    
    instruction_file = "fast_user_instruction.json" if args.fast_simulation else "reason_user_instruction.json"
    with open(f"{args.simulation_prompt_dir}/{instruction_file}", "r") as f:
        instruction = json.load(f)
    
    return data, system_message, instruction

def simulate_opinions(llm, tokenizer, topic_data, state, persona_type, persona_dir, args, system_message, instruction):
    """Run simulation for a specific topic and state"""
    simulation_message = generate_topics(topic_data)
    return generate_opinions_batch(
        llm=llm,
        tokenizer=tokenizer,
        simulation_message=simulation_message,
        system_message=system_message,
        topic_data=topic_data,
        persona_type=persona_type,
        persona_dir=persona_dir,
        state=state,
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        instruction=instruction
    )

def save_simulation_results(results, topic_index, output_dir, meta_stats, model_info, state):
    """Save simulation results and update meta stats with topic-first hierarchical structure"""
    os.makedirs(output_dir, exist_ok=True)
    output_file = f"{output_dir}/topic_{topic_index}_opinions.json"
    with open(output_file, "w") as f:
        json.dump(results, f, indent=2)

    # Initialize topic if it doesn't exist
    topic_key = f"topic_{topic_index}"
    if topic_key not in meta_stats:
        meta_stats[topic_key] = {
            "question_id": results["topic"],
            "Topic": results["topic"],
            "models": {}
        }
    
    # Create a unique model identifier
    model_key = f"{model_info['persona_model']}_{model_info['persona_type']}_{model_info['simulation_model']}"
    
    # Initialize model info if it doesn't exist
    if model_key not in meta_stats[topic_key]["models"]:
        meta_stats[topic_key]["models"][model_key] = {
            "model_info": model_info,
            "states": {}
        }
    
    # Add state-specific results
    meta_stats[topic_key]["models"][model_key]["states"][state] = {
        "counts": results["counts"],
        "percentage": results["percentage"]
    }

def main():
    parser = argparse.ArgumentParser(description="Inference")
    parser.add_argument("--model_backend", required=True, help="Model backend")
    parser.add_argument("--model_name", required=True, help="Simulation model name")
    parser.add_argument("--HF_HOME_DIR", required=True, help="HF home directory")
    parser.add_argument("--persona_model_name_list", required=True, type=lambda x: ast.literal_eval(x.strip('"')), help="Persona model name")
    parser.add_argument("--simulation_model_name", required=True, help="Simulation model name")
    parser.add_argument("--persona_type_list", required=True, type=lambda x: ast.literal_eval(x.strip('"')), help="Persona type")
    parser.add_argument("--topic_list", required=True, type=lambda x: ast.literal_eval(x.strip('"')), help="Topic")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Top p")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max tokens")
    parser.add_argument("--gpu_memory_utilization", type=float, default=0.95, required=False, help="GPU memory utilization (default: 0.95)")
    parser.add_argument("--persona_dir", required=True, help="Persona directory")
    parser.add_argument("--meta_persona_dir", required=True, help="Meta persona directory")
    parser.add_argument("--topic_dir", required=True, help="Topic directory")
    parser.add_argument("--simulation_prompt_dir", required=True, help="Simulation prompt directory")
    parser.add_argument("--simulation_result_dir", required=True, help="Simulation result directory")   
    parser.add_argument("--meta_stats_dir", required=True, help="Meta stats directory")
    parser.add_argument("--fast_simulation", type=lambda x: x.lower() == 'true', default=False, help="Use fast user instruction (default: False)")
    args = parser.parse_args()

    # Initialize model
    llm, tokenizer = setup_model(args)
    if not llm or not tokenizer:
        return

    # Get list of states (excluding AK, HI)
    STATES = [state.abbr for state in us.states.STATES if state.abbr not in ['AK', 'HI']]

    for topic in args.topic_list:
        print(f'Simulating for {topic}')
        data, system_message, instruction = load_topic_resources(topic, args)
        meta_stats = {}  # Moved meta_stats outside the persona_type loop

        for persona_type in args.persona_type_list:
            print(f'Simulating for {persona_type}')

            for state in STATES:
                # for topic_index in range(1, len(data) + 1):
                for topic_index in range(1, 100):
                    print(f'Generating for {state} - topic{topic_index}')
                    topic_data = data[f"topic_{topic_index}"]

                    if "meta" in persona_type:
                        persona_dir = args.meta_persona_dir.format(state=state)
                        results = simulate_opinions(llm, tokenizer, topic_data, state, 
                                                 persona_type, persona_dir, args, 
                                                 system_message, instruction)
                        
                        output_dir = args.simulation_result_dir.format(
                            Topic=topic, 
                            persona_model_name="meta",
                            simulation_model_name=args.simulation_model_name,
                            state=state, 
                            persona_type=persona_type,
                            fast_simulation=args.fast_simulation
                        )
                        model_info = {
                            "persona_model": "meta",
                            "persona_type": persona_type,
                            "simulation_model": args.simulation_model_name
                        }
                        save_simulation_results(results, topic_index, output_dir, meta_stats, model_info, state)
                    
                    else:
                        for persona_model_name in args.persona_model_name_list:
                            print(f'Simulating for {persona_model_name}')
                            persona_dir = args.persona_dir.format(
                                persona_model_name=persona_model_name,
                                persona_type=persona_type,
                                state=state
                            )
                            results = simulate_opinions(llm, tokenizer, topic_data, state, 
                                                     persona_type, persona_dir, args, 
                                                     system_message, instruction)
                            
                            output_dir = args.simulation_result_dir.format(
                                Topic=topic,
                                persona_model_name=persona_model_name,
                                simulation_model_name=args.simulation_model_name,
                                state=state,
                                persona_type=persona_type,
                                fast_simulation=args.fast_simulation
                            )
                            model_info = {
                                "persona_model": persona_model_name,
                                "persona_type": persona_type,
                                "simulation_model": args.simulation_model_name
                            }
                            save_simulation_results(results, topic_index, output_dir, meta_stats, model_info, state)

        # Save meta stats for the state
        meta_stats_file = f"{args.meta_stats_dir.format(Topic=topic)}/meta_stats.json"
        with open(meta_stats_file, "w") as f:
            json.dump(meta_stats, f, indent=2)
        print(f"Meta stats saved to {meta_stats_file}")


if __name__ == "__main__":
    main()