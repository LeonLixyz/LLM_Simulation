
import torch
import json
from collections import namedtuple

# create a namedspace for Distribution, where we have both the
CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_relationship_distribution(data, total_population):
    relationship = data['RELATIONSHIP']
    house_meta_categories = [
        'Homeless',
        'Primary Householder',
        'Spouse of Householder',
        'Child living with Parents',
        'Other Relative of Householder',
        'Non-Relative Housemate'
    ]

    house_meta_probabilities = torch.tensor([
        (total_population - relationship['Total']) / total_population,
        relationship['Householder']['count'] / total_population,
        relationship['Spouse']['count'] / total_population,
        relationship['Child']['count'] / total_population,
        relationship['Other relatives']['count'] / total_population,
        relationship['Other nonrelatives']['count'] / total_population
    ])

    return CategoricalDistribution(categories=house_meta_categories, 
                                   distribution=torch.distributions.Categorical(probs=house_meta_probabilities))

def sample_relationship(dp02_data, dp05_data):
    relationship_dist = create_relationship_distribution(dp02_data, dp05_data["SEX AND AGE"]["Total population"])
    return relationship_dist.categories[relationship_dist.distribution.sample()]

def create_household_distribution(household_data):
    household_categories = ['Married-couple', 'Cohabiting couple', 'Single']
    household_probabilities = torch.tensor([
        household_data['Family households']['Married-couple']['percentage'],
        household_data['Family households']['Cohabiting couple']['percentage'],
        household_data['Nonfamily households']['Male householder']['percentage'] + 
        household_data['Nonfamily households']['Female householder']['percentage']
    ])
    return CategoricalDistribution(categories=household_categories, 
                                   distribution=torch.distributions.Categorical(probs=household_probabilities))

def create_single_distribution(household_data):
    single_categories = ['Male', 'Female']
    single_probabilities = torch.tensor([
        household_data['Nonfamily households']['Male householder']['percentage'],
        household_data['Nonfamily households']['Female householder']['percentage']
    ])
    return CategoricalDistribution(categories=single_categories, 
                                   distribution=torch.distributions.Categorical(probs=single_probabilities))

def create_kids_distribution(household_data, household_type):
    if household_type == 'Married-couple':
        with_kids = household_data['Family households']['Married-couple']['With children under 18']['percentage']
        without_kids = household_data['Family households']['Married-couple']['percentage'] - with_kids
    elif household_type == 'Cohabiting couple':
        with_kids = household_data['Family households']['Cohabiting couple']['With children under 18']['percentage']
        without_kids = household_data['Family households']['Cohabiting couple']['percentage'] - with_kids
    elif household_type == 'Single Male':
        with_kids = household_data['Nonfamily households']['Male householder']['With children under 18']['percentage']
        without_kids = household_data['Nonfamily households']['Male householder']['percentage'] - with_kids
    elif household_type == 'Single Female':
        with_kids = household_data['Nonfamily households']['Female householder']['With children under 18']['percentage']
        without_kids = household_data['Nonfamily households']['Female householder']['percentage'] - with_kids
    else:
        raise ValueError(f"Unknown household type: {household_type}")
    
    probabilities = torch.tensor([with_kids, without_kids])
    return CategoricalDistribution(categories=['With kids', 'Without kids'], 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_household_type(data, gender=None):
    household_data = data['HOUSEHOLDS BY TYPE']
    household_dist = create_household_distribution(household_data)
    household_type = household_dist.categories[household_dist.distribution.sample()]
    
    if household_type == 'Single':
        if gender:
            single_type = gender
        else:
            single_dist = create_single_distribution(household_data)
            single_type = single_dist.categories[single_dist.distribution.sample()]
        household_type = f"Single {single_type}"
    
    kids_dist = create_kids_distribution(household_data, household_type)
    has_kids = kids_dist.categories[kids_dist.distribution.sample()]
    
    return f"{household_type} {has_kids}"