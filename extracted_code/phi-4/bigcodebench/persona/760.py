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
    
    # Create a list of all names
    names = latin_names + other_names
    
    # Generate random IDs
    ids = np.arange(1, 101)
    
    # Generate random names
    selected_names = np.random.choice(names, size=100)
    
    # Correct improperly encoded Latin characters
    def correct_encoding(name):
        try:
            return codecs.decode(name, 'utf-8')
        except (UnicodeDecodeError, TypeError):
            return name
    
    corrected_names = [correct_encoding(name) for name in selected_names]
    
    # Generate random dates of birth
    date_of_birth = pd.date_range(start=f'{start_year}-01-01', end=f'{end_year}-12-31', periods=100).to_pydatetime()
    np.random.shuffle(date_of_birth)
    
    # Generate emails
    emails = [f"{name.lower().replace(' ', '').replace('ñ', 'n')}{''.join(re.findall(r'\d+', str(date.year)))}@{email_domain}"
              for name, date in zip(corrected_names, date_of_birth)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': ids,
        'Name': corrected_names,
        'Date of Birth': date_of_birth,
        'Email': emails
    })
    
    return df