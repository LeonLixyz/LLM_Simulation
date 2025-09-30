import torch
from collections import namedtuple, Counter

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_education_distribution(education_data):
    categories = [
        'Less than 9th grade',
        '9th to 12th grade, no diploma',
        'High school graduate',
        'Some college, no degree',
        "Associate's degree",
        "Bachelor's degree",
        'Graduate or professional degree'
    ]
    probabilities = torch.tensor([
        education_data['Less than 9th grade']['percentage'],
        education_data['9th to 12th grade, no diploma']['percentage'],
        education_data['High school graduate']['percentage'],
        education_data['Some college, no degree']['percentage'],
        education_data["Associate's degree"]['percentage'],
        education_data["Bachelor's degree"]['percentage'],
        education_data['Graduate or professional degree']['percentage']
    ]) / 100  # Convert percentages to probabilities
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_education_level(data):
    education_data = data['EDUCATIONAL ATTAINMENT']
    education_distribution = create_education_distribution(education_data)
    return education_distribution.categories[education_distribution.distribution.sample()]