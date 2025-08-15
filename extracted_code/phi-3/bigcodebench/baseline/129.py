import requests
from bs4 import BeautifulSoup
import pandas as pd

def task_func(url='http://example.com'):
    try:
        response = requests.get(url)
        response.raise_for_status()
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
    except requests.exceptions.ConnectionError as err:
        raise SystemExit(err)
    try:
        soup = BeautifulSoup(response.content, 'html.parser')
        table = soup.find('table')
        if not table:
            raise ValueError("No table data found on the page.")
        headers = [th.get_text(strip=True) for th in table.find_all('th')]
        rows = table.find_all('tr')
        data = [[td.get_text(strip=True) for td in row.find_all('td')] for row in rows[1:]]
        df = pd.DataFrame(data, columns=headers)
        return df
    except Exception as e:
        raise SystemExit(e)