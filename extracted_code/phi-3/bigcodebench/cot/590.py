import urllib.request
from pyquery import PyQuery as pq
from datetime import datetime
import pandas as pd

def task_func(url):
    if not url or not isinstance(url, str):
        raise ValueError("Invalid or empty URL provided.")
    
    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read()
            doc = pq(html_content)
            anchor_data = [
                {
                    'text': anchor.text(),
                    'href': anchor.attr('href'),
                    'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
                }
                for anchor in doc('a')
            ]
            return pd.DataFrame(anchor_data)
    except urllib.error.URLError as e:
        raise Exception(f"URL Error: {e.reason}")