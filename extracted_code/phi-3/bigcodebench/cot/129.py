import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    
    soup = BeautifulSoup(response.text, 'html.parser')
    table = soup.find('table')
    
    if not table:
        raise ValueError('No table found on the page.')
    
    headers = [th.get_text(strip=True) for th in table.find_all('th')]
    
    rows = []
    for tr in table.find_all('tr'):
        td_values = [td.get_text(strip=True) for td in tr.find_all('td')]
        if td_values:
            rows.append(td_values)
    
    if not rows:
        raise ValueError('No table data found.')

    return pd.DataFrame(rows, columns=headers)

# Example usage:
# df = task_func('https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)')
# print(df)