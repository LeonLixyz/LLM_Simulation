#!/bin/bash
#SBATCH -p columbia        # partition name
#SBATCH -N 1              # number of nodes
#SBATCH -c 96             # number of cores
#SBATCH --mem=2000000     # memory in MB (2TB)
#SBATCH --gpus=8          # number of GPUs
#SBATCH -J persona_gen    # job name
#SBATCH -o slurm-%j.out   # output file
#SBATCH -e slurm-%j.err   # error file

# Activate your environment
conda activate vllm_env

# Run your script
python run_persona_gen.py