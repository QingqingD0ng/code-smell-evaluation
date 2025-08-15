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
        table = tree.xpath('//table')
        
        if not table:
            return 0

        df = pd.read_html(html.tostring(table[0]))[0]

        if df.empty:
            return 0

        conn = sqlite3.connect(database_name)
        cursor = conn.cursor()

        cursor.execute("DROP TABLE IF EXISTS my_table")
        conn.commit()

        df.to_sql("my_table", conn, if_exists='replace', index=False)
        conn.commit()

        cursor.execute("SELECT COUNT(*) FROM my_table")
        row_count = cursor.fetchone()[0]

        conn.close()
        return row_count

    except requests.RequestException:
        raise
    except sqlite3.DatabaseError:
        raise