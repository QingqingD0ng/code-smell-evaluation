import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    try:
        if webpage_url.startswith("file://"):
            with open(webpage_url[7:], 'r') as file:
                content = file.read()
        else:
            response = requests.get(webpage_url)
            response.raise_for_status()
            content = response.text

        tree = html.fromstring(content)
        table = tree.xpath('//table')[0]  # Assuming there's at least one table
        df = pd.read_html(html.tostring(table), flavor='lxml')[0]

        if df.empty:
            return 0

        conn = sqlite3.connect(database_name)
        df.to_sql('my_table', conn, if_exists='replace', index=False)
        conn.close()

        return len(df)

    except requests.RequestException:
        raise
    except sqlite3.DatabaseError:
        raise