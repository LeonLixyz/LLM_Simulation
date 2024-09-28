import numpy as np
import torch
from collections import namedtuple

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_insurance_distribution(data, employment_status):
    categories = ['Private health insurance', 'Public coverage', 'No health insurance coverage']
    
    if employment_status == 'Employed':
        private = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['With health insurance coverage']['Employed']['Private health insurance']['percentage']
        public = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['With health insurance coverage']['Employed']['Public coverage']['percentage']
        none = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['No health insurance coverage']['Employed']['percentage']
    elif employment_status == 'Unemployed':
        private = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['With health insurance coverage']['Unemployed']['Private health insurance']['percentage']
        public = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['With health insurance coverage']['Unemployed']['Public coverage']['percentage']
        none = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['No health insurance coverage']['Unemployed']['percentage']
    else:  # Not in labor force
        private = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['With health insurance coverage']['Not in labor force']['Private health insurance']['percentage']
        public = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['With health insurance coverage']['Not in labor force']['Public coverage']['percentage']
        none = data['HEALTH INSURANCE COVERAGE']['Civilian noninstitutionalized population 19 to 64 years']['No health insurance coverage']['Not in labor force']['percentage']
    
    probabilities = torch.tensor([private, public, none]) / 100
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def create_income_distribution(data):
    categories = [
        'Less than $10,000', '$10,000 to $14,999', '$15,000 to $24,999',
        '$25,000 to $34,999', '$35,000 to $49,999', '$50,000 to $74,999',
        '$75,000 to $99,999', '$100,000 to $149,999', '$150,000 to $199,999',
        '$200,000 or more'
    ]
    probabilities = torch.tensor([
        data['INCOME AND BENEFITS'][cat]['percentage'] for cat in categories
    ]) / 100
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_insurance_and_income(row, data):
    employment_status = row['EMPLOYMENT STATUS']
    labor_force_status = row['LABOR FORCE STATUS']
    
    if labor_force_status == 'Not in Labor Force':
        insurance_dist = create_insurance_distribution(data, 'Not in labor force')
    else:
        insurance_dist = create_insurance_distribution(data, employment_status)

    insurance = insurance_dist.categories[insurance_dist.distribution.sample()]
    
    if employment_status == 'Employed':
        income_dist = create_income_distribution(data)
        income = income_dist.categories[income_dist.distribution.sample()]
    else:
        income = None
    
    return insurance, income