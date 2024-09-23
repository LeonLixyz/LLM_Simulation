import os
import json
import pandas as pd
from tqdm import tqdm
import us
from get_dist.get_sex_race_age_dist import load_structured_data, get_sex_dist, get_age_dist, get_race_dist, sample_demographics
from get_dist.get_household_dist import sample_household_type, sample_relationship
from get_dist.get_marital_status_dist import sample_marital_status_by_gender
from get_dist.get_veteran_dist import sample_veteran_status
from get_dist.get_language_dist import sample_languages
from get_dist.get_edu_dist import sample_education_level
from get_dist.get_birth_dist import sample_birth_and_citizenship_multiple

def generate_personas_for_state(state):
    # Load data
    dp05 = f"/user/al4263/Simulate/Persona/data/DP05/structured_data/{state.abbr.lower()}_structured_data.json"
    dp02 = f"/user/al4263/Simulate/Persona/data/DP02/structured_data/{state.abbr.lower()}_dp02_structured.json"

    dp05_data = load_structured_data(dp05)
    dp02_data = load_structured_data(dp02)

    # Get distributions
    age_distributions = get_age_dist(dp05_data)
    sex_distributions = get_sex_dist(dp05_data)
    race_distributions = get_race_dist(dp05_data)

    # Sample demographics
    num_samples = 1000
    samples = [sample_demographics(age_distributions, sex_distributions, race_distributions) for _ in range(num_samples)]

    # Create DataFrame
    df = pd.DataFrame(samples)

    # Add HOUSEHOLD_RELATIONSHIP
    df['HOUSEHOLD_RELATIONSHIP'] = df.apply(lambda row: sample_relationship(dp02_data, dp05_data), axis=1)

    # Sample household type for primary householders
    def sample_household_type_for_primary(row):
        if row['HOUSEHOLD_RELATIONSHIP'] == 'Primary Householder':
            return sample_household_type(dp02_data, row['Sex'])
        return None

    df['HOUSEHOLD_TYPE'] = df.apply(sample_household_type_for_primary, axis=1)

    # Sample additional attributes
    df['MARITAL_STATUS'] = df.apply(lambda row: sample_marital_status_by_gender(dp02_data, row['Sex']), axis=1)
    df['VETERAN_STATUS'] = df.apply(lambda row: sample_veteran_status(dp02_data), axis=1)
    df['LANGUAGE'], df['ENGLISH_PROFICIENCY'] = zip(*df.apply(lambda row: sample_languages(dp02_data), axis=1))
    df['EDUCATION'] = df.apply(lambda row: sample_education_level(dp02_data), axis=1)
    df['BIRTH_PLACE'], df['CITIZENSHIP'], df['BIRTH_DETAIL'] = zip(*df.apply(lambda row: sample_birth_and_citizenship_multiple(dp02_data), axis=1))

    # Add state information
    df['STATE_NAME'] = state.name
    df['STATE_ABBR'] = state.abbr

    # Save personas
    base_dir = f"/user/al4263/Simulate/Persona/data/persona_meta/{state.abbr}"
    os.makedirs(base_dir, exist_ok=True)

    for index, row in df.iterrows():
        file_name = f"{state.abbr}_persona_{index}.json"
        file_path = os.path.join(base_dir, file_name)
        
        with open(file_path, 'w') as f:
            json.dump(row.to_dict(), f, indent=2)

    print(f"Saved {len(df)} persona files for {state.name} in {base_dir}")

if __name__ == "__main__":
    # Generate personas for all states
    for state in tqdm(us.states.STATES, desc="Generating personas for states"):
        generate_personas_for_state(state)

    print("Persona generation complete for all states.")