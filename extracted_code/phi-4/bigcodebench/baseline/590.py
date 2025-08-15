import urllib.request
from pyquery import PyQuery as pq
from datetime import datetime
import pandas as pd
from urllib.error import URLError

def task_func(url):
    if not url:
        raise ValueError("URL cannot be empty")

    try:
        with urllib.request.urlopen(url) as response:
            html_content = response.read()

        doc = pq(html_content)
        anchors = doc('a')

        data = []
        for anchor in anchors:
            text = pq(anchor).text()
            href = pq(anchor).attr('href')
            data.append({'text': text, 'href': href})

        fetch_time = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
        df = pd.DataFrame(data)
        df['fetch_time'] = fetch_time

        return df

    except URLError as e:
        raise URLError("Network issue or server error") from e