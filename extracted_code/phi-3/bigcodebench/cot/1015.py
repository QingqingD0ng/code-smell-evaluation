import requests
from lxml import html
import pandas as pd
import sqlite3

def task_func(webpage_url: str, database_name: str = "my_database.db") -> int:
    try:
        response = requests.get(webpage_url)
        response.raise_for_status()
        html_content = response.text
    except requests.RequestException as e:
        raise e

    try:
        tree = html.fromstring(html_content)
        data_table = tree.xpath('//table//tr')

        df = pd.DataFrame([row.xpath('.//td//text()') for row in data_table if row.xpath('.//td//text()')])
        df.columns = df.iloc[0].values
        df = df[1:]

        df.dropna(inplace=True)

        conn = sqlite3.connect(database_name)
        df.to_sql('my_table', conn, if_exists='replace', index=False)
        conn.commit()
    except Exception as e:
        raise e

    finally:
        conn.close()

    return len(df)

if __name__ == "__main__":
    try:
        num_rows = task_func("http://example.com/tabledata")
        print(f"Number of rows parsed: {num_rows}")
    except Exception as e:
        print(f"An error occurred: {e}")