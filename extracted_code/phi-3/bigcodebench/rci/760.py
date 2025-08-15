import pandas as pd
import numpy as np
import codecs
import re
import datetime

def task_func(start_year=1980, end_year=2000, email_domain='example.com',
               latin_names=['Sopetón', 'Méndez', 'Gómez', 'Pérez', 'Muñoz'],
               other_names=['Smith', 'Johnson', 'Williams', 'Brown', 'Jones'], 
               rng_seed=None):
    # Set the random seed for reproducibility
    np.random.seed(rng_seed)

    # Constants
    NUM_RECORDS = 100
    ID_RANGE = (1, 100)
    LATIN_NAMES = latin_names
    OTHER_NAMES = other_names
    EMAIL_DOMAIN = email_domain

    # Generate unique IDs
    ids = np.arange(*ID_RANGE)

    # Randomly select names
    names = np.random.choice(LATIN_NAMES + OTHER_NAMES, size=NUM_RECORDS, replace=True)

    # Generate random birth years
    birth_years = np.random.randint(start_year, end_year + 1, size=NUM_RECORDS)

    # Generate random birth dates
    birth_months = np.random.randint(1, 13, size=NUM_RECORDS)
    birth_days = np.random.randint(1, 29, size=NUM_RECORDS)  # Simplified, not accounting for leap years
    dates_of_birth = [datetime.date(year, month, day) for year, month, day in zip(birth_years, birth_months, birth_days)]

    # Generate emails
    emails = [f"{re.sub(r'[^a-zA-Z0-9]', '', name.lower())}{year}{EMAIL_DOMAIN}" for name, year in zip(names, birth_years)]

    # Create