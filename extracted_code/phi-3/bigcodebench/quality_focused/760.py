import pandas as pd
import numpy as np
import codecs
import re
import datetime

def task_func(start_year=1980, end_year=2000, email_domain='example.com',
             latin_names=['Sopetón', 'Méndez', 'Gómez', 'Pérez', 'Muñoz'],
             other_names=['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], 
             rng_seed=None):
    
    np.random.seed(rng_seed)
    
    ids = np.arange(1, 101)
    names = np.random.choice(latin_names + other_names, size=100, replace=True)
    birth_years = np.random.randint(start_year, end_year + 1, size=100)
    dates_of_birth = [datetime.date(year, 12 + np.random.randint(3, 12), np.random.randint(1, 31)) for year in birth_years]
    emails = [f"{name}{birth_years[i]}{email_domain}" for i, name in enumerate(names)]
    
    df = pd.DataFrame({
        'ID': ids,
        'Name': names,
        'Date of Birth': dates_of_birth,
        'Email': emails
    })
    
    # Correct improperly encoded Latin characters
    df['Name'] = df['Name'].apply(lambda x: x.encode('latin1').decode('utf-8'))
    
    return df