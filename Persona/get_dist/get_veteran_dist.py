import torch
from collections import namedtuple, Counter

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_veteran_status_distribution(veteran_data):
    categories = ['Veteran', 'Non-Veteran']
    probabilities = torch.tensor([
        veteran_data['Veteran']['percentage'],
        100 - veteran_data['Veteran']['percentage']
    ]) / 100  # Convert percentages to probabilities
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_veteran_status(data):
    veteran_distribution = create_veteran_status_distribution(data['VETERAN STATUS'])
    return veteran_distribution.categories[veteran_distribution.distribution.sample()]

