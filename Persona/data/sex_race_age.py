import requests
import pandas as pd
import us

def get_state_data(state_fips):
    data_url = f"https://api.census.gov/data/2023/acs/acs1/profile?get=group(DP05)&for=state:{state_fips}"
    response = requests.get(data_url)
    return response.json()

metadata_url = "https://api.census.gov/data/2023/acs/acs1/profile/variables.json"
metadata_response = requests.get(metadata_url)
variables = metadata_response.json()['variables']

dp05_mapping = {}
for var_code, var_info in variables.items():
    if var_code.startswith('DP05'):  
        dp05_mapping[var_code] = var_info['label']

for state in us.states.STATES:
    print(f"Processing {state.name}...")
    
    raw_data = get_state_data(state.fips)
    
    variable_codes = raw_data[0]
    values = raw_data[1]

    data_df = pd.DataFrame({
        "Variable Code": variable_codes,
        "Value": values
    })

    data_df['Label'] = data_df['Variable Code'].map(dp05_mapping)
    data_df = data_df[['Label', 'Value']].dropna()
    
    # Save the data with state-specific filename
    filename = f"{state.abbr.lower()}_sex_race_age.csv"
    data_df.to_csv(filename, index=False)
    print(f"Saved {filename}")

print("All states processed!")