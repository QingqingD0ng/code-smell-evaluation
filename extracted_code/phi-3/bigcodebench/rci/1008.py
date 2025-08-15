import requests
from bs4 import BeautifulSoup
import pandas as pd
from io import StringIO

def fetch_webpage_soup(url):
    try:
        response = requests.get(url)
        response.raise_for_status()
        return BeautifulSoup(response.content, 'html.parser')
    except requests.exceptions.RequestException as err:
        raise SystemExit(err)

def extract_table_data(soup, table_id):
    table = soup.find("table", {"id": table_id})
    if table is None:
        raise ValueError(f"No table found with the specified ID '{table_id}'.")
    
    rows = table.find_all("tr")
    if not rows:
        return pd.DataFrame()
    
    headers = [header.text.strip() for header in rows[0].find_all("th")]
    data = [[col.text.strip() for col in row.find_all("td")] for row in rows[1:]]
    
    return pd.DataFrame(data, columns=headers)

def task_func(url, table_id):
    try:
        soup = fetch_webpage_soup(url)
        df = extract_table_data(soup, table_id)
        return df
    except ValueError as err:
        print(err)
        return pd.DataFrame()
    except SystemExit as err:
        print(err)
        return pd.DataFrame()

# Example usage:
# df = task_func('https://example.com/data.html', 'table1')
# print(df)