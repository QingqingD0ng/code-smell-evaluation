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
    dates_of_birth = [datetime.date(year, np.random.randint(1, 12), np.random.randint(1, 28)) for year in birth_years]
    emails = [f"{re.sub(r'[^a-zA-Z0-9]', '', name.lower())}{year}{email_domain}" for name, year in zip(names, birth_years)]
    
    df = pd.DataFrame({
        'ID': ids,
        'Name': names,
        'Date of Birth': dates_of_birth,
        'Email': emails
    })
    
    for name in latin_names:
        df['Name'] = df['Name'].str.replace(codecs.BOM_UTF8.decode('utf-8-sig'), name.encode('utf-8'))
    
    return df

# Example usage:
df = task_func(rng_seed=1)
print(df)

df = task_func(start_year=0, end_year=1200, email_domain='test.at', rng_seed=3)
print(df)