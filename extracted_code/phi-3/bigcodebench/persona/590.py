import urllib.request
from pyquery import PyQuery as pq
import datetime
import pandas as pd

def task_func(url):
    if not url:
        raise ValueError("URL cannot be empty")
    try:
        with urllib.request.urlopen(url) as response:
            html = response.read()
        doc = pq(html)
        data = []
        for anchor in doc('a').items():
            data.append({
                'text': anchor.text().strip(),
                'href': anchor.attr('href').strip(),
                'fetch_time': datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')
            })
        return pd.DataFrame(data)
    except urllib.error.URLError as e:
        raise urllib.error.URLError(f"URL Error: {e.reason}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")