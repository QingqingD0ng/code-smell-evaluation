import urllib.request
from pyquery import PyQuery as pq
from datetime import datetime
import pandas as pd

def task_func(url):
    if not url:
        raise ValueError("URL cannot be empty")
    try:
        response = urllib.request.urlopen(url)
        html = response.read()
    except urllib.error.URLError as e:
        raise e
    doc = pq(html)
    data = []
    for anchor in doc('a').items():
        row = {
            'text': anchor.text(),
            'href': anchor.attr('href'),
            'fetch_time': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        }
        data.append(row)
    df = pd.DataFrame(data)
    return df