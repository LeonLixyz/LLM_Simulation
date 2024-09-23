
import torch
from collections import namedtuple

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_marital_status_distribution(marital_data):
    categories = ['Never married', 'Now married, except separated', 'Separated', 'Widowed', 'Divorced']
    probabilities = torch.tensor([
        marital_data['Never married']['percentage'],
        marital_data['Now married, except separated']['percentage'],
        marital_data['Separated']['percentage'],
        marital_data['Widowed']['percentage'],
        marital_data['Divorced']['percentage']
    ]) / 100  # Convert percentages to probabilities
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_marital_status(marital_distribution):
    return marital_distribution.categories[marital_distribution.distribution.sample()]

def create_marital_status_distributions(dp02_data):
    marital_data = dp02_data['MARITAL STATUS']
    male_distribution = create_marital_status_distribution(marital_data['Males'])
    female_distribution = create_marital_status_distribution(marital_data['Females'])
    return {'Male': male_distribution, 'Female': female_distribution}

def sample_marital_status_by_gender(data, gender):
    marital_distributions = create_marital_status_distributions(data)
    gender = gender.capitalize()  # Ensure correct capitalization
    if gender not in ['Male', 'Female']:
        raise ValueError("Gender must be 'Male' or 'Female'")
    return sample_marital_status(marital_distributions[gender])