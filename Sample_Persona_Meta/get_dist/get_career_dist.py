import torch
from get_dist.get_sex_race_age_dist import load_structured_data
from collections import namedtuple
import random

CategoricalDistribution = namedtuple('CategoricalDistribution', ['categories', 'distribution'])

def create_distribution(data):
    total = sum(data.values())
    categories = list(data.keys())
    probabilities = torch.tensor([value / total for value in data.values()])
    return CategoricalDistribution(categories=categories, 
                                   distribution=torch.distributions.Categorical(probs=probabilities))

def sample_career(gender_data):
    # Sample main category
    main_categories = {
        "Management, business, science, and arts occupations": gender_data["Management, business, science, and arts occupations"]["Total"],
        "Service occupations": gender_data["Service occupations"]["Total"],
        "Sales and office occupations": gender_data["Sales and office occupations"]["Total"],
        "Natural resources, construction, and maintenance occupations": gender_data["Natural resources, construction, and maintenance occupations"]["Total"],
        "Production, transportation, and material moving occupations": gender_data["Production, transportation, and material moving occupations"]["Total"]
    }
    main_dist = create_distribution(main_categories)
    main_category = main_dist.categories[main_dist.distribution.sample()]

    if main_category == "Management, business, science, and arts occupations":
        mbsa_data = gender_data[main_category]["Subcategories"]
        mbsa_categories = {k: v["Total"] for k, v in mbsa_data.items()}
        mbsa_dist = create_distribution(mbsa_categories)
        mbsa_subcategory = mbsa_dist.categories[mbsa_dist.distribution.sample()]

        if mbsa_subcategory == "Management, business, and financial occupations":
            mbf_data = mbsa_data[mbsa_subcategory]["Subcategories"]
            mbf_dist = create_distribution(mbf_data)
            return mbf_dist.categories[mbf_dist.distribution.sample()]
        elif mbsa_subcategory == "Computer, engineering, and science occupations":
            ces_data = mbsa_data[mbsa_subcategory]["Subcategories"]
            ces_dist = create_distribution(ces_data)
            return ces_dist.categories[ces_dist.distribution.sample()]
        elif mbsa_subcategory == "Education, legal, community service, arts, and media occupations":
            elcsam_data = mbsa_data[mbsa_subcategory]["Subcategories"]
            elcsam_dist = create_distribution(elcsam_data)
            return elcsam_dist.categories[elcsam_dist.distribution.sample()]
        elif mbsa_subcategory == "Healthcare practitioner and technical occupations":
            hpt_data = mbsa_data[mbsa_subcategory]["Subcategories"]
            hpt_dist = create_distribution(hpt_data)
            return hpt_dist.categories[hpt_dist.distribution.sample()]

    elif main_category == "Service occupations":
        service_data = gender_data[main_category]["Subcategories"]
        service_categories = {k: v["Total"] for k, v in service_data.items()}
        service_dist = create_distribution(service_categories)
        service_subcategory = service_dist.categories[service_dist.distribution.sample()]

        if service_subcategory == "Protective service occupations":
            ps_data = service_data[service_subcategory]["Subcategories"]
            ps_dist = create_distribution(ps_data)
            return ps_dist.categories[ps_dist.distribution.sample()]
        else:
            return service_subcategory

    elif main_category == "Sales and office occupations":
        so_data = gender_data[main_category]["Subcategories"]
        so_categories = {k: v["Total"] for k, v in so_data.items()}
        so_dist = create_distribution(so_categories)
        return so_dist.categories[so_dist.distribution.sample()]

    elif main_category == "Natural resources, construction, and maintenance occupations":
        nrcm_data = gender_data[main_category]["Subcategories"]
        nrcm_categories = {k: v["Total"] for k, v in nrcm_data.items()}
        nrcm_dist = create_distribution(nrcm_categories)
        return nrcm_dist.categories[nrcm_dist.distribution.sample()]

    elif main_category == "Production, transportation, and material moving occupations":
        ptmm_data = gender_data[main_category]["Subcategories"]
        ptmm_categories = {k: v["Total"] for k, v in ptmm_data.items()}
        ptmm_dist = create_distribution(ptmm_categories)
        return ptmm_dist.categories[ptmm_dist.distribution.sample()]

def generate_career(row, occupation_data):
    if row['EMPLOYMENT STATUS'] == 'Employed':
        return sample_career(occupation_data[row['SEX']])
    elif row['AGE'] in ['18 to 19 years', '20 to 24 years'] and row['EMPLOYMENT STATUS'] != 'Employed':
        if random.random() < 0.7:  # 70% chance
            return "Student"
        else:
            return None
    else:
        return None