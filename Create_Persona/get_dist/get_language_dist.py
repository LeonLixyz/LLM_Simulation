import torch
from collections import namedtuple, Counter

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_language_distribution(language_data):
    categories = [
        'English only',
        'Spanish',
        'Other Indo-European languages',
        'Asian and Pacific Islander languages',
        'Other languages'
    ]
    probabilities = torch.tensor([
        language_data['English only']['percentage'],
        language_data['Spanish']['percentage'],
        language_data['Other Indo-European languages']['percentage'],
        language_data['Asian and Pacific Islander languages']['percentage'],
        language_data['Other languages']['percentage']
    ]) / 100  # Convert percentages to probabilities
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def create_english_proficiency_distribution(language_data, language):
    speak_well = language_data[language]['percentage'] - language_data[language]['Speak English less than very well']['percentage']
    speak_less_well = language_data[language]['Speak English less than very well']['percentage']
    probabilities = torch.tensor([speak_well, speak_less_well]) / (speak_well + speak_less_well)
    return CategoricalDistribution(categories=['Speak English well', 'Speak English less than very well'],
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_language(language_distribution):
    return language_distribution.categories[language_distribution.distribution.sample()]

def sample_english_proficiency(proficiency_distribution):
    return proficiency_distribution.categories[proficiency_distribution.distribution.sample()]

def sample_languages(data):
    language_data = data['LANGUAGE SPOKEN AT HOME']
    language_distribution = create_language_distribution(language_data)
    proficiency_distributions = {
        lang: create_english_proficiency_distribution(language_data, lang)
        for lang in ['Spanish', 'Other Indo-European languages', 'Asian and Pacific Islander languages', 'Other languages']
    }
    
    language = sample_language(language_distribution)
    if language == 'English only':
        return (language, "Speak English well")
    else:
        proficiency = sample_english_proficiency(proficiency_distributions[language])
        return (language, proficiency)