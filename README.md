# LLM_Simulation

## Running Simulation

To generate persona and run simulations, follow these steps:

1. Generate persona metadata:
   Run the `generate_persona_meta` function to create initial persona data.

2. Generate full personas:
   Use the `opinion_simulation.ipynb` notebook to generate complete personas and run simulations.

## Project Structure

- `data/`: Contains various data files used in the project
  - `persona_meta/`: Stores generated persona metadata
  - `persona_llama_generation/`: Stores full persona data generated using LLaMA
  - `prompts/`: Contains prompt templates used for persona generation and simulations
    - `election/`: Contains simulation questions regarding election
    - `opinion_simulation/`: Contains prompts for simulating opinions.

- `get_dist/`: Contains probability sampling functions for different meta demographic information.

- `opinion_simulation.ipynb`: Main notebook for generating personas and running simulations
