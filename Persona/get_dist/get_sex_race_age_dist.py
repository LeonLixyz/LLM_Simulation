import torch
import json
from collections import namedtuple

# create a namedspace for Distribution, where we have both the
CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

RaceDistributions = namedtuple('RaceDistributions', [
    'main_race',
    'american_indian_alaska_native',
    'asian',
    'native_hawaiian_pacific_islander',
    'two_or_more_races',
    'hispanic',
    'hispanic_subcategories'
])

Demographics = namedtuple('Demographics', ['Age', 'Sex', 'Race', 'Ethnicity'])



def load_structured_data(dir):
    # load the structured data from the directory, this is a absolute path
    with open(dir, 'r') as file:
        structured_data = json.load(file)
    return structured_data


def get_sex_dist(data):
    sex_data = data['SEX AND AGE']

    male_percentage = sex_data['18 years and over Male']['percentage']
    female_percentage = sex_data['18 years and over Female']['percentage']

    # Create sex probabilities
    sex_probabilities = [male_percentage / 100.0, female_percentage / 100.0]
    sex_categories = ['Male', 'Female']
    sex_probs_tensor = torch.tensor(sex_probabilities)

    # Create Categorical distribution for Sex
    sex_distribution = torch.distributions.Categorical(probs=sex_probs_tensor)

    return CategoricalDistribution(categories=sex_categories, distribution=sex_distribution)

def get_age_dist(data):
    age_data = data['SEX AND AGE']['Age Groups']
    age_categories = list(age_data.keys())
    age_probabilities = [age_data[category]['percentage'] / 100.0 for category in age_categories]
    age_probs_tensor = torch.tensor(age_probabilities)

    # Create Categorical distribution for Age
    age_distribution = torch.distributions.Categorical(probs=age_probs_tensor)

    return CategoricalDistribution(categories=age_categories, distribution=age_distribution)

def get_race_dist(data):
    race_data = data['RACE']
    main_race_categories = ['White', 'Black or African American', 'American Indian and Alaska Native', 'Asian', 'Native Hawaiian and Other Pacific Islander', 'Some Other Race', 'Two or More Races']

    race_counts = []
    race_percentages = []
    race_categories = []

    for category in main_race_categories:
        if category in race_data:
            count = race_data[category]['count']
            percentage = race_data[category]['percentage']
            race_counts.append(count)
            race_percentages.append(percentage)
            race_categories.append(category)
        else:
            print(f"Category {category} not found in data")

    # Normalize race percentages to sum to 1
    total_race_percentage = sum(race_percentages)
    race_probabilities = [p / total_race_percentage for p in race_percentages]
    race_probs_tensor = torch.tensor(race_probabilities)

    # Create Categorical distribution for Race
    race_distribution = torch.distributions.Categorical(probs=race_probs_tensor)

    # Process subcategories for races with ethnicities
    # For 'American Indian and Alaska Native'
    ai_an_ethnicities = race_data['American Indian and Alaska Native']['Ethnicities']

    ai_an_ethnicities_list = []
    ai_an_percentages = []

    for ethnicity, values in ai_an_ethnicities.items():
        count = values['count']
        percentage = values['percentage']
        ai_an_ethnicities_list.append(ethnicity)
        ai_an_percentages.append(percentage)

    # Normalize percentages
    total_ai_an_percentage = sum(ai_an_percentages)
    ai_an_probabilities = [p / total_ai_an_percentage for p in ai_an_percentages]
    ai_an_probs_tensor = torch.tensor(ai_an_probabilities)

    # Create Categorical distribution
    ai_an_distribution = torch.distributions.Categorical(probs=ai_an_probs_tensor)

    # For 'Asian'
    asian_ethnicities = race_data['Asian']['Ethnicities']

    asian_ethnicities_list = []
    asian_percentages = []

    for ethnicity, values in asian_ethnicities.items():
        count = values['count']
        percentage = values['percentage']
        asian_ethnicities_list.append(ethnicity)
        asian_percentages.append(percentage)

    # Normalize percentages
    total_asian_percentage = sum(asian_percentages)
    asian_probabilities = [p / total_asian_percentage for p in asian_percentages]
    asian_probs_tensor = torch.tensor(asian_probabilities)

    # Create Categorical distribution
    asian_distribution = torch.distributions.Categorical(probs=asian_probs_tensor)

    # For 'Native Hawaiian and Other Pacific Islander'
    nhopi_ethnicities = race_data['Native Hawaiian and Other Pacific Islander']['Ethnicities']

    nhopi_ethnicities_list = []
    nhopi_percentages = []

    for ethnicity, values in nhopi_ethnicities.items():
        count = values['count']
        percentage = values['percentage']
        nhopi_ethnicities_list.append(ethnicity)
        nhopi_percentages.append(percentage)

    # Normalize percentages
    total_nhopi_percentage = sum(nhopi_percentages)
    nhopi_probabilities = [p / total_nhopi_percentage for p in nhopi_percentages]
    nhopi_probs_tensor = torch.tensor(nhopi_probabilities)

    # Create Categorical distribution
    nhopi_distribution = torch.distributions.Categorical(probs=nhopi_probs_tensor)

    # For 'Two or More Races'
    two_or_more_races = race_data['Two or More Races']['Combinations']

    combinations_list = []
    combinations_percentages = []

    for combination, values in two_or_more_races.items():
        count = values['count']
        percentage = values['percentage']
        combinations_list.append(combination)
        combinations_percentages.append(percentage)

    # Normalize percentages
    total_combinations_percentage = sum(combinations_percentages)
    combinations_probabilities = [p / total_combinations_percentage for p in combinations_percentages]
    combinations_probs_tensor = torch.tensor(combinations_probabilities)

    # Create Categorical distribution
    combinations_distribution = torch.distributions.Categorical(probs=combinations_probs_tensor)

    # Process 'HISPANIC OR LATINO AND RACE'
    hispanic_data = data['HISPANIC OR LATINO AND RACE']
    hispanic_categories = ['Hispanic or Latino (of any race)', 'Not Hispanic or Latino']

    hispanic_counts = []
    hispanic_percentages = []
    hispanic_categories_list = []

    for category in hispanic_categories:
        if category in hispanic_data:
            count = hispanic_data[category]['count']
            percentage = hispanic_data[category]['percentage']
            hispanic_counts.append(count)
            hispanic_percentages.append(percentage)
            hispanic_categories_list.append(category)

    # Normalize percentages
    total_hispanic_percentage = sum(hispanic_percentages)
    hispanic_probabilities = [p / total_hispanic_percentage for p in hispanic_percentages]
    hispanic_probs_tensor = torch.tensor(hispanic_probabilities)

    # Create Categorical distribution
    hispanic_distribution = torch.distributions.Categorical(probs=hispanic_probs_tensor)

    # Process 'Hispanic or Latino' subcategories
    hispanic_subcategories = hispanic_data['Hispanic or Latino']

    hispanic_subcategories_list = []
    hispanic_sub_percentages = []

    for subcategory, values in hispanic_subcategories.items():
        if subcategory != 'count' and subcategory != 'percentage':
            count = values['count']
            percentage = values['percentage']
            hispanic_subcategories_list.append(subcategory)
            hispanic_sub_percentages.append(percentage)

    # Normalize percentages
    total_hispanic_sub_percentage = sum(hispanic_sub_percentages)
    hispanic_sub_probabilities = [p / total_hispanic_sub_percentage for p in hispanic_sub_percentages]
    hispanic_sub_probs_tensor = torch.tensor(hispanic_sub_probabilities)

    # Create Categorical distribution
    hispanic_sub_distribution = torch.distributions.Categorical(probs=hispanic_sub_probs_tensor)

    return RaceDistributions(
        main_race=CategoricalDistribution(categories=race_categories, distribution=race_distribution),
        american_indian_alaska_native=CategoricalDistribution(categories=ai_an_ethnicities_list, distribution=ai_an_distribution),
        asian=CategoricalDistribution(categories=asian_ethnicities_list, distribution=asian_distribution),
        native_hawaiian_pacific_islander=CategoricalDistribution(categories=nhopi_ethnicities_list, distribution=nhopi_distribution),
        two_or_more_races=CategoricalDistribution(categories=combinations_list, distribution=combinations_distribution),
        hispanic=CategoricalDistribution(categories=hispanic_categories_list, distribution=hispanic_distribution),
        hispanic_subcategories=CategoricalDistribution(categories=hispanic_subcategories_list, distribution=hispanic_sub_distribution)
    )

def sample_demographics(age_distributions, sex_distributions, race_distributions):

    race_distribution = race_distributions.main_race.distribution
    race_categories = race_distributions.main_race.categories
    hispanic_distribution = race_distributions.hispanic.distribution
    hispanic_categories_list = race_distributions.hispanic.categories
    hispanic_sub_distribution = race_distributions.hispanic_subcategories.distribution
    hispanic_subcategories_list = race_distributions.hispanic_subcategories.categories
    ai_an_distribution = race_distributions.american_indian_alaska_native.distribution
    ai_an_ethnicities_list = race_distributions.american_indian_alaska_native.categories
    asian_distribution = race_distributions.asian.distribution
    asian_ethnicities_list = race_distributions.asian.categories
    nhopi_distribution = race_distributions.native_hawaiian_pacific_islander.distribution
    nhopi_ethnicities_list = race_distributions.native_hawaiian_pacific_islander.categories
    combinations_distribution = race_distributions.two_or_more_races.distribution
    combinations_list = race_distributions.two_or_more_races.categories
    age_distribution = age_distributions.distribution
    age_categories = age_distributions.categories
    sex_distribution = sex_distributions.distribution
    sex_categories = sex_distributions.categories

    hispanic_status = hispanic_categories_list[hispanic_distribution.sample().item()]
    
    race = None
    ethnicity = None
    
    if hispanic_status == 'Hispanic or Latino (of any race)':
        hispanic_subtype = hispanic_subcategories_list[hispanic_sub_distribution.sample().item()]
        race = 'Hispanic'
        ethnicity = hispanic_subtype
    else:
        race = race_categories[race_distribution.sample().item()]
        
        if race == 'American Indian and Alaska Native':
            ethnicity = ai_an_ethnicities_list[ai_an_distribution.sample().item()]
        elif race == 'Asian':
            ethnicity = asian_ethnicities_list[asian_distribution.sample().item()]
        elif race == 'Native Hawaiian and Other Pacific Islander':
            ethnicity = nhopi_ethnicities_list[nhopi_distribution.sample().item()]
        elif race == 'Two or More Races':
            race = combinations_list[combinations_distribution.sample().item()]
            ethnicity = "None"
    
    age_group = age_categories[age_distribution.sample().item()]
    sex = sex_categories[sex_distribution.sample().item()]

    return Demographics(Age=age_group, Sex=sex, Race=race, Ethnicity=ethnicity)

