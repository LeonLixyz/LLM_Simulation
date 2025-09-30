#!/bin/bash
#SBATCH -p columbia
#SBATCH -N 1
#SBATCH -c 96
#SBATCH --mem=2000000
#SBATCH --gpus=8
#SBATCH --job-name=sim_Athene-70B
#SBATCH --output=./LLM_Simulation/slurm_run_logs/sim_Athene-70B.out

source ~/.bashrc
conda activate vllm_env
python Opinion_Simulation.py --temperature 0.95 --top_p 0.9 --max_tokens 512 --persona_dir ./LLM_Simulation/Persona/{persona_model_name}/{state}/{persona_type} --meta_persona_dir ./LLM_Simulation/Persona/Meta_Persona/{state}/meta_persona --topic_dir ./LLM_Simulation/Simulation_Result/{Topic} --simulation_prompt_dir ./LLM_Simulation/Prompts/Opinion_Simulation --simulation_result_dir ./LLM_Simulation/Simulation_Result/{Topic}/Sim_{simulation_model_name}/P_{persona_model_name}/{state}-FAST{fast_simulation}/{persona_type} --fast_simulation True --gpu_memory_utilization 0.95 --topic_list "['Election']" --persona_type_list "['objective_table_persona','subjective_table_persona','descriptive_persona']" --persona_model_name_list "['Llama-3.1-8B-Instruct','Llama-3.1-70B-Instruct','Athene-70B','Mixtral-8x7B-Instruct-v0.1','Nemotron-70B-Instruct','Qwen2.5-72B-Instruct']" --model_backend vllm --model_name Nexusflow/Athene-70B --simulation_model_name Athene-70B --HF_HOME_DIR ./huggingface/cache
