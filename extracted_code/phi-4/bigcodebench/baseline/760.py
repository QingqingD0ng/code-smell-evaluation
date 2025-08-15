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
    
    ids = np.arange(1, 101)
    names = latin_names + other_names
    random_names = np.random.choice(names, 100)
    
    # Generate random dates
    start_date = datetime.date(start_year, 1, 1)
    end_date = datetime.date(end_year, 12, 31)
    time_between_dates = end_date - start_date
    days_between_dates = time_between_dates.days
    random_days = np.random.randint(0, days_between_dates, 100)
    random_dates = [start_date + datetime.timedelta(days=x) for x in random_days]
    
    # Fix encoding for Latin characters
    def fix_encoding(name):
        return codecs.decode(codecs.encode(name, 'latin1'), 'utf8','replace')
    
    fixed_names = [fix_encoding(name) for name in random_names]
    
    # Construct emails
    emails = [f"{name.lower()}{date.year}@{email_domain}" for name, date in zip(fixed_names, random_dates)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'Name': fixed_names,
        'Date of Birth': random_dates,
        'Email': emails
    })
    
    return df