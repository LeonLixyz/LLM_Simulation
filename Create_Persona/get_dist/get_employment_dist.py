import numpy as np
import torch
from collections import namedtuple

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_labor_force_distribution(stats):
    categories = ['In Labor Force', 'Not in Labor Force']
    labor_force_rate = stats['Labor Force Participation Rate'] / 100
    probabilities = torch.tensor([labor_force_rate, 1 - labor_force_rate])
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def create_employment_distribution(stats):
    categories = ['Employed', 'Unemployed']
    employment_rate = 1 - (stats['Unemployment rate'] / 100)
    probabilities = torch.tensor([employment_rate, 1 - employment_rate])
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_labor_force_and_employment(age_group, employment_data):
    # Map the age groups from df to the ones in employment_data
    age_group_mapping = {
        '18 to 19 years': '16 to 19 years',
        '20 to 24 years': '20 to 24 years',
        '25 to 34 years': '25 to 29 years',  # Use the first of the two relevant groups
        '35 to 44 years': '35 to 44 years',
        '45 to 54 years': '45 to 54 years',
        '55 to 59 years': '55 to 59 years',
        '60 to 64 years': '60 to 64 years',
        '65 to 74 years': '65 to 74 years',
        '75 to 84 years': '75 years and over',
        '85 years and over': '75 years and over'
    }
    
    mapped_age_group = age_group_mapping.get(age_group, age_group)
    
    stats = employment_data['EMPLOYMENT STATISTICS']['AGE'][mapped_age_group]
    
    labor_force_dist = create_labor_force_distribution(stats)
    employment_dist = create_employment_distribution(stats)
    
    labor_force_status = labor_force_dist.categories[labor_force_dist.distribution.sample()]
    
    if labor_force_status == 'In Labor Force':
        employment_status = employment_dist.categories[employment_dist.distribution.sample()]
    else:
        employment_status = 'Not Applicable'
    
    return labor_force_status, employment_status
