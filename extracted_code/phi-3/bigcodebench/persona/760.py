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
    
    # Generate random dates of birth
    birth_dates = [datetime.date(np.random.randint(start_year, end_year + 1), 
                                  np.random.randint(1, 12), 
                                  np.random.randint(1, 28)) for _ in range(100)]
    
    # Randomly select names from Latin and other names
    names = np.random.choice(latin_names + other_names, 100, p=[0.5]*len(latin_names) + [0.5]*len(other_names))
    
    # Construct email addresses
    emails = [f"{name}{birth_date.year}{email_domain}" for name, birth_date in zip(names, birth_dates)]
    
    # Create DataFrame
    df = pd.DataFrame({
        'ID': np.arange(1, 101),
        'Name': names,
        'Date of Birth': birth_dates,
        'Email': emails
    })
    
    # Correct improperly encoded Latin characters
    df['Name'] = df['Name'].apply(lambda x: codecs.encode(x, 'unicode_escape').decode('utf-8'))
    
    return df

if __name__ == "__main__":
    df = task_func(rng_seed=1)
    print(df)