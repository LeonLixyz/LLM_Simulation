import os
import json
import random
from transformers import AutoTokenizer
import us
from tqdm import tqdm
import argparse
from vllm import LLM, SamplingParams

# Set up cache directories
HF_HOME_DIR = "./huggingface/cache"
TRANSFORMERS_CACHE_DIR = "./huggingface/cache"
os.makedirs(HF_HOME_DIR, exist_ok=True)
os.makedirs(TRANSFORMERS_CACHE_DIR, exist_ok=True)

os.environ['HF_HOME'] = HF_HOME_DIR
os.environ['TRANSFORMERS_CACHE'] = TRANSFORMERS_CACHE_DIR
# os.environ['CUDA_VISIBLE_DEVICES'] = '0,1,3,4,5,6,7'

def load_llm(model_name, gpu_memory_utilization):
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
    
def load_meta_persona(meta_data):
    persona_info =  meta_data.get("persona") or meta_data.get("PERSONA") or meta_data
    persona_info = {
        "AGE": persona_info["AGE"],
        "SEX": persona_info["SEX"],
        "RACE": persona_info["RACE"],
        "EDUCATION": persona_info["EDUCATION"],
        "HOUSEHOLD_RELATIONSHIP": persona_info["HOUSEHOLD_RELATIONSHIP"],
        "MARITAL_STATUS": persona_info["MARITAL_STATUS"],
        "VETERAN_STATUS": persona_info["VETERAN_STATUS"],
        "CAREER": persona_info["CAREER"],
        "INCOME_RANGE": persona_info["INCOME_RANGE"],
        "INSURANCE_COVERAGE": persona_info["INSURANCE_COVERAGE"],
        "STATE_NAME": persona_info["STATE_NAME"],
        "STATE_ABBR": persona_info["STATE_ABBR"]
    }
    return persona_info

def calcualte_total_counts(persona_state, num_samples):
    meta_persona_dir = f"./LLM_Simulation/Persona/Persona_State_Ultimate/{persona_state}/marginal_persona"
    personas = []
    for i in range(num_samples):
        with open(os.path.join(meta_persona_dir, f"persona_{i}.json"), 'r') as f:
            persona = json.load(f)['PERSONA']
            personas.append(persona)

    selected_features = ['EDUCATION', 'HOUSEHOLD_RELATIONSHIP', 'MARITAL_STATUS', 'VETERAN_STATUS', 'CAREER', 'INCOME_RANGE', 'INSURANCE_COVERAGE']

    feature_counts = {feature: {} for feature in selected_features}

    for persona in personas:
        for feature, value in persona.items():
            if feature in selected_features:
                if isinstance(value, dict):
                    for sub_feature, sub_value in value.items():
                        if sub_feature not in feature_counts[feature]:
                            feature_counts[feature][sub_feature] = {}
                        if sub_value not in feature_counts[feature][sub_feature]:
                            feature_counts[feature][sub_feature][sub_value] = 0
                        feature_counts[feature][sub_feature][sub_value] += 1
                else:
                    if value not in feature_counts[feature]:
                        feature_counts[feature][value] = 0
                    feature_counts[feature][value] += 1

    return feature_counts

def resample_features(persona_list, feature_counts):
    # Resample features for a list of personas
    resampled_personas = []
    features_to_resample = ['EDUCATION', 'HOUSEHOLD_RELATIONSHIP', 'MARITAL_STATUS', 'VETERAN_STATUS', 'CAREER', 'INCOME_RANGE', 'INSURANCE_COVERAGE']
    for persona in persona_list:
        new_persona = persona.copy()
        for feature in features_to_resample:
            counts = feature_counts[feature]
            total = sum(counts.values())
            if total == 0:
                raise ValueError(f"No remaining counts for feature {feature}")
            # Create a list of possible values and their probabilities
            values = list(counts.keys())
            probabilities = [counts[value] / total for value in values]
            # Sample a value
            sampled_value = random.choices(values, weights=probabilities, k=1)[0]
            # Update the persona
            new_persona[feature] = sampled_value
        resampled_personas.append(new_persona)
    return resampled_personas

def update_feature_counts(feature_counts, persona):
    # Decrement the counts for the features used in the persona
    features_to_update = ['EDUCATION', 'HOUSEHOLD_RELATIONSHIP', 'MARITAL_STATUS', 'VETERAN_STATUS', 'CAREER', 'INCOME_RANGE', 'INSURANCE_COVERAGE']
    for feature in features_to_update:
        value = persona[feature]
        feature_counts[feature][value] -= 1
        if feature_counts[feature][value] < 1:
            # always preserve some probability to smooth the sampling
            feature_counts[feature][value] = 1
    return feature_counts

def verify_and_resample_personas_batch(llm, tokenizer, num_personas, start_index, temperature, top_p, max_tokens, persona_type, persona_state, batch_size, max_iterations=100):
    base_dir = f"./LLM_Simulation/Persona/Persona_State_Ultimate/{persona_state}/marginal_persona"
    with open(f"/user/al4263/Simulate/Prompts/General/persona_generation/marginal_persona_calibration/system.json", "r") as f:
        system_message = json.load(f)["system_message"]
    
    with open(f"/user/al4263/Simulate/Prompts/General/persona_generation/marginal_persona_calibration/instructions.json", "r") as f:
        instructions = json.load(f)["instructions"]

    # Load total counts and initialize feature counts
    total_counts = calcualte_total_counts(persona_state, num_personas)
    feature_counts = {feature: counts.copy() for feature, counts in total_counts.items()}

    if num_personas == -1:
        num_personas = len([f for f in os.listdir(base_dir) if f.endswith('.json')])

    # Load all personas
    all_personas = []
    for i in range(start_index, num_personas):
        file_name = f"persona_{i}.json"
        file_path = os.path.join(base_dir, file_name)
        with open(file_path, "r") as f:
            metadata = json.load(f)
        mini_persona = load_meta_persona(metadata)
        all_personas.append((i, mini_persona))

    iteration = 0
    remaining_personas = all_personas
    while iteration < max_iterations and remaining_personas:
        print(f"Iteration {iteration+1}, processing {len(remaining_personas)} personas")
        batch_prompts = []
        batch_indices = []
        sampling_params_list = []

        # Process in batches
        for batch_start in range(0, len(remaining_personas), batch_size):
            batch_end = min(batch_start + batch_size, len(remaining_personas))
            batch = remaining_personas[batch_start:batch_end]
            batch_prompts = []
            batch_indices = []
            sampling_params_list = []

            for index, persona in batch:
                user_prompt = instructions.format(METADATA=json.dumps(persona, indent=2))
                messages = [
                    {"role": "system", "content": system_message},
                    {"role": "user", "content": user_prompt}
                ]
                prompt = tokenizer.apply_chat_template(messages, tokenize=False)
                batch_prompts.append(prompt)
                batch_indices.append(index)

                sampling_params = SamplingParams(
                    temperature=temperature,
                    top_p=top_p,
                    max_tokens=max_tokens,
                    seed=index + iteration  
                )
                sampling_params_list.append(sampling_params)

            outputs = llm.generate(batch_prompts, sampling_params_list)
            counts = {"correct": 0, "incorrect": 0}
            counts_meta = {}
            next_iteration_personas = []
            for output, (index, persona) in zip(outputs, batch):
                verification_result = extract_verification_result(output.outputs[0].text)
                if verification_result == "1":
                    counts["correct"] += 1
                    # Update feature counts
                    feature_counts = update_feature_counts(feature_counts, persona)
                    # Save the correct persona
                    save_persona(index, persona, persona_type, persona_state)
                else:
                    counts["incorrect"] += 1
                    # Collect personas for next iteration
                    next_iteration_personas.append((index, persona))
                counts_meta[index] = verification_result
            print(f'Verification counts for iteration {iteration+1}: {counts}')
            # counts_meta["meta_stats"] = counts
            # save_counts(counts_meta, persona_type, persona_state, iteration, batch_start)

            # Prepare for next batch
            remaining_personas = next_iteration_personas

        # Resample features for incorrect personas
        if remaining_personas:
            print(f"Resampling {len(remaining_personas)} incorrect personas")
            # Resample features for all incorrect personas
            personas_to_resample = [persona for index, persona in remaining_personas]
            resampled_personas = resample_features(personas_to_resample, feature_counts)
            # Prepare for next iteration
            remaining_personas = list(zip([index for index, persona in remaining_personas], resampled_personas))
        else:
            break  # All personas are correct

        iteration += 1

    if remaining_personas:
        print(f"After {max_iterations} iterations, {len(remaining_personas)} personas could not be corrected.")
        # for index, persona in remaining_personas:
        #     save_persona(index, persona, persona_type, persona_state)

def extract_verification_result(output):
    persona_parts = output.split("Answer:")
    if len(persona_parts) > 1:
        return persona_parts[1].strip()
    else:
        output = output.replace("<|start_header_id|>assistant<|end_header_id|>\n\n", "")
        return output.strip()

def save_persona(index, persona, persona_type, persona_state):
    base_dir = f"./LLM_Simulation/Persona/Persona_State_Ultimate/{persona_state}/{persona_type}"
    os.makedirs(base_dir, exist_ok=True)
    
    file_name = f"persona_{index}.json"
    file_path = os.path.join(base_dir, file_name)

    data = {"PERSONA": persona}
    
    with open(file_path, "w") as file:
        json.dump(data, file, indent=2)

def main():
    parser = argparse.ArgumentParser(description="Persona Verification and Resampling")
    parser.add_argument("--model_backend", required=True, help="Model backend")
    parser.add_argument("--model_name", required=True, help="Model name")
    parser.add_argument("--persona_type", required=True, help="Persona type")
    parser.add_argument("--num_personas", type=int, required=True, help="Number of personas")
    parser.add_argument("--start_index", type=int, required=True, help="Start index")
    parser.add_argument("--temperature", type=float, required=True, help="Temperature")
    parser.add_argument("--top_p", type=float, required=True, help="Top p")
    parser.add_argument("--max_tokens", type=int, required=True, help="Max tokens")
    parser.add_argument("--batch_size", type=int, required=False, default=8, help="Batch size")
    parser.add_argument("--gpu_memory_utilization", type=float, required=False, default=0.95, help="GPU memory utilization")
    args = parser.parse_args()

    if args.model_backend == "vllm":
        llm = load_llm(args.model_name, args.gpu_memory_utilization)
        tokenizer = load_tokenizer(args.model_name)

    PERSONA_STATES = [state.abbr for state in us.states.STATES]
    # PERSONA_STATES = ["NY"]
    for persona_state in PERSONA_STATES:
        print(f'Processing for {persona_state}')
        verify_and_resample_personas_batch(llm, tokenizer, args.num_personas, args.start_index, 
                                           args.temperature, args.top_p, args.max_tokens, 
                                           args.persona_type, persona_state, args.batch_size)

    print("Persona verification and resampling complete.")

if __name__ == "__main__":
    main()
