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
        
    def encode_name(name):
        encoded = codecs.encode(name, 'latin-1', 'ignore')
        return encoded.decode('latin-1')
    
    all_names = latin_names + other_names
    ids = np.arange(1, 101)
    names = np.random.choice(all_names, size=100)
    names = [encode_name(name) for name in names]
    dates_of_birth = [
        datetime.date(
            np.random.randint(start_year, end_year + 1),
            np.random.randint(1, 13),
            np.random.randint(1, 29)
        )
        for _ in range(100)
    ]
    emails = [
        f"{name.lower()}{date.year}@{email_domain}"
        for name, date in zip(names, dates_of_birth)
    ]
    
    data = {
        'ID': ids,
        'Name': names,
        'Date of Birth': dates_of_birth,
        'Email': emails
    }
    
    return pd.DataFrame(data)