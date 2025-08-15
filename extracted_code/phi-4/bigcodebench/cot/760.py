import pandas as pd
import numpy as np
import codecs
import re
import datetime

def task_func(start_year=1980, end_year=2000, email_domain='example.com',
              latin_names=['Sopetón', 'Méndez', 'Gómez', 'Pérez', 'Muñoz'],
              other_names=['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], 
              rng_seed=None):
    
    if rng_seed is not None:
        np.random.seed(rng_seed)
    
    all_names = latin_names + other_names
    ids = np.arange(1, 101)
    names = np.random.choice(all_names, size=100)
    random_days = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31').to_numpy()
    birth_dates = np.random.choice(random_days, size=100)

    def clean_name(name):
        name = name.encode('latin1').decode('utf-8', errors='ignore')
        name = re.sub(r'\W+', '', name)
        return name

    cleaned_names = np.array([clean_name(name) for name in names])
    years_of_birth = birth_dates.astype(str).str[:4]
    emails = np.core.defchararray.add(
        np.core.defchararray.add(cleaned_names, years_of_birth), f'@{email_domain}')

    df = pd.DataFrame({
        'ID': ids,
        'Name': names,
        'Date of Birth': birth_dates,
        'Email': emails
    })

    return df