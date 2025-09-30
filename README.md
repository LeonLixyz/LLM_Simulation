# Persona is a promise with a catch

## Project Structure

- `Sample_Persona_Meta/`: Code for generating meta personas from US Census Bureau data
  - `get_dist/`: Probability sampling functions for demographic information (age, sex, race, education, income, etc.)
  - `collect_data/`: Data collection notebooks from US Census Bureau
  - `sample_marginal.py`: Main script for generating meta personas

- `Gen_Sim/`: Core simulation framework
  - `Persona_Generation.py`: Generates detailed personas from meta personas
  - `Opinion_Simulation.py`: Runs opinion simulations with generated personas
  - `Persona_Calibration.py`: Calibrating persona generation to avoid sampling bias.
  - `configs/`: Configuration files for different models and simulation parameters
  - `model_configs/`: Model-specific configurations for various LLMs
  - `run_persona_gen.py`: Script for generating personas
  - `run_simluation_topic.py`: Script for running topic-based simulations
  - `run_simluation_slurm.py`: SLURM cluster execution script

- `Prompts/`: Prompt templates for persona generation and opinion simulation
  - `Persona_Generation/`: Prompts for creating detailed personas. We also include prompts we used for experimentation. For the simulation experiments in the paper, we used the following persona types:
    - `objective_table_persona/`: Structured objective persona with demographic tables
    - `subjective_table_persona/`: Structured subjective persona with opinion tables  
    - `descriptive_persona/`: Natural language descriptive personas
  - `Opinion_Simulation/`: Prompts for opinion simulation tasks
    - `fast_user_instruction.json`: Fast simulation without chain-of-thought
    - `multi_turn_*.json`: Multi-turn conversation simulation prompts
    - `reason_user_instruction.json`: CoT-based opinion generation

- `Simulation_Result/`: Output directory for simulation results
  - Contains results organized by topic and persona type

## Workflow

### Step 1: Generate Meta Personas
First, generate meta personas from US Census data:

```bash
cd Sample_Persona_Meta
python sample_marginal.py
```

This creates demographic samples for all US states based on Census Bureau data.

### Step 2: Generate Detailed Personas
Generate detailed personas using LLMs:

```bash
cd Gen_Sim
python run_persona_gen.py
```

This creates the three persona types (objective table, subjective table, descriptive) for each state.

### Step 3: Run Opinion Simulations
First, create simulation questions in the `Simulation_Result/` directory, register the questions as a file called `context.json`. See the `context.json` file in the `Simulation_Result/` directory for an example.

Run opinion simulations on various topics:

```bash
cd Gen_Sim
python run_simluation_topic.py
```

This simulates opinions across different topics using the generated personas.

## Configuration

- `configs/persona_config.json`: Configuration for persona generation
- `configs/simulation_config.json`: Configuration for opinion simulations
- `model_configs/`: Model-specific configurations for different LLMs

## Models we used

- Llama-3.1-8B-Instruct
- Llama-3.1-70B-Instruct  
- Athene-70B
- Mixtral-8x7B-Instruct-v0.1
- Nemotron-70B-Instruct
- Qwen2.5-72B-Instruct

You can register new models in the `model_configs/` directory.




