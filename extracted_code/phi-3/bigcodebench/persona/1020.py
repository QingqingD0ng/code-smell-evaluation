import json
import requests
import chardet

API_URL = "http://api.example.com/data"

def task_func(url=API_URL, from_encoding=None, to_encoding="utf8"):
    response = requests.get(url, timeout=5)
    content_type = response.headers.get('Content-Type')
    charset = None

    if 'charset=' in content_type:
        charset = content_type.split('charset=')[-1]
    elif from_encoding is not None:
        charset = from_encoding
    else:
        charset = chardet.detect(response.content)['encoding']

    if not response.content and charset:
        return {}

    if not charset:
        raise ValueError("Unable to detect encoding for non-empty content")

    raw_content = response.content.decode(charset)
    json_content = json.loads(raw_content)

    if json_content:
        return json_content

    return {}