import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        print(f"HTTP error occurred: {err}")
        return None
    except requests.exceptions.ConnectionError as err:
        print(f"Connection error occurred: {err}")
        return None

    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if not table:
            print("No table data found on the page.")
            return None

        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = table.find_all('tr')
        data = [[td.get_text(strip=True) for td in row.find_all('td')] for row in rows[1:]]
        df = pd.DataFrame(data, columns=headers)
        return df
    except Exception as e:
        print(f"An error occurred while parsing the table data: {e}")
        return None

# Test the function
df = task_func('https://en.wikipedia.org/wiki/List_of_countries_by_GDP_(nominal)')
if df is not None:
    print(df)