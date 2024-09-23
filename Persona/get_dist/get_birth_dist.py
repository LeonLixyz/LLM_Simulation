import torch
from collections import namedtuple, Counter

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_birth_place_distribution(birth_data):
    categories = ['US Born', 'Foreign Born']
    probabilities = torch.tensor([
        birth_data['Native']['percentage'],
        birth_data['Foreign born']['percentage']
    ]) / 100  # Convert percentages to probabilities
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def create_us_born_distribution(birth_data):
    native_data = birth_data['Native']
    total_native = native_data['count']
    categories = ['State of residence', 'Different state', 'US territories or abroad to American parents']
    probabilities = torch.tensor([
        native_data['State of residence']['count'] / total_native,
        native_data['Different state']['count'] / total_native,
        native_data['Born in Puerto Rico, U.S. Island areas, or born abroad to American parent(s)   ']['count'] / total_native
    ])
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def create_foreign_born_distribution(birth_data):
    foreign_data = birth_data['Foreign born']
    categories = ['US Citizen', 'Not a U.S. citizen']
    probabilities = torch.tensor([
        foreign_data['US Citizen']['percentage'],
        foreign_data['Not a U.S. citizen']['percentage']
    ]) / 100  # Convert percentages to probabilities
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def create_foreign_born_region_distribution(birth_data):
    foreign_data = birth_data['Foreign born']
    categories = ['Europe', 'Asia', 'Africa', 'Oceania', 'Latin America', 'Northern America']
    probabilities = torch.tensor([foreign_data[cat]['percentage'] for cat in categories]) / 100
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_birth_and_citizenship(birth_data):
    birth_place_dist = create_birth_place_distribution(birth_data)
    us_born_dist = create_us_born_distribution(birth_data)
    foreign_born_dist = create_foreign_born_distribution(birth_data)
    foreign_region_dist = create_foreign_born_region_distribution(birth_data)

    birth_place = sample_distribution(birth_place_dist)
    
    if birth_place == 'US Born':
        us_birth_place = sample_distribution(us_born_dist)
        return {'Birth Place': 'US Born', 'Specific': us_birth_place, 'Citizenship': 'US Citizen'}
    else:
        citizenship = sample_distribution(foreign_born_dist)
        region = sample_distribution(foreign_region_dist)
        return {'Birth Place': 'Foreign Born', 'Citizenship': citizenship, 'Region': region}

def sample_distribution(distribution):
    return distribution.categories[distribution.distribution.sample()]

def sample_birth_and_citizenship_multiple(data):
    birth_data = data['PLACE OF BIRTH']
    result = sample_birth_and_citizenship(birth_data)
    birth_place = result['Birth Place']
    citizenship = result['Citizenship']
    specific_or_region = result.get('Specific') or result.get('Region')

    return birth_place, citizenship, specific_or_region


