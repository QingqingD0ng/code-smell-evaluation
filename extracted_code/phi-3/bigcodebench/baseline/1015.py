import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
        tree = html.fromstring(response.content)
        table = tree.xpath('//table')[0]
        df = pd.read_html(html.tostring(table))[0]
        if df.empty:
            return 0
        conn = sqlite3.connect(database_name)
        df.to_sql('my_table', con=conn, if_exists='replace', index=False)
        return df.shape[0]
    except requests.RequestException as e:
        raise e
    except sqlite3.DatabaseError as e:
        raise e