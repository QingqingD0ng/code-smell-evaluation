import pandas as pd
import numpy as np
import re
from datetime import datetime, timedelta

# Constants for name lists
latin_names = ['Sopetón', 'Méndez', 'Gómez', 'Pérez', 'Muñoz']
other_names = ['Smith', 'Johnson', 'Williams', 'Brown', 'Jones']

def task_func(start_year=1980, end_year=2000, email_domain='example.com',
              latin_names=latin_names, other_names=other_names, rng_seed=None):
    if rng_seed is not None:
        np.random.seed(rng_seed)

    # Generate random IDs, birth years, names, and dates of birth
    ids = np.arange(1, 101)
    birth_years = np.random.randint(start_year, end_year + 1, size=100)
    names = np.random.choice(latin_names + other_names, size=100)
    dates_of_birth = [datetime(year, np.random.randint(1, 13), np.random.randint(1, 29)).strftime('%Y-%m-%d') for year in birth_years]
    emails = [f"{name}{birth_years[i]}{email_domain}" for i, name in enumerate(names)]

    # Create DataFrame
    df = pd.DataFrame({'ID': ids, 'Name': names, 'Date of Birth': dates_of_birth, 'Email': emails})

    # Correct improperly encoded Latin characters in names
    df['Name'] = df['Name'].apply(lambda x: re.sub(r'[^a-zA-ZáéíóúÁÉÍÓÚñÑ]', '', x))
    
    return df