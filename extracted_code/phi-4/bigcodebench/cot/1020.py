import json
import requests
import chardet

def task_func(url="http://api.example.com/data", from_encoding=None, to_encoding="utf-8"):
    try:
        response = requests.get(url, timeout=5)
    except requests.RequestException:
        return {}

    if not response.content:
        return {}

    if from_encoding is None:
        detected = chardet.detect(response.content)
        from_encoding = detected['encoding']
        if not from_encoding:
            raise ValueError("Unable to detect encoding for non-empty content")

    decoded_content = response.content.decode(from_encoding)
    re_encoded_content = decoded_content.encode(to_encoding)
    return json.loads(re_encoded_content.decode(to_encoding))