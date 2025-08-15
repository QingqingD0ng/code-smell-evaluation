import json
import requests
import chardet

API_URL = "http://api.example.com/data"

def task_func(url=API_URL, from_encoding=None, to_encoding="utf8"):
    response = requests.get(url, timeout=5)
    content = response.content
    
    if not content:
        return {}
    
    if from_encoding is None:
        detected = chardet.detect(content)
        if not detected['confidence']:
            raise ValueError("Unable to detect encoding for non-empty content")
        from_encoding = detected['encoding']
    
    decoded_content = content.decode(from_encoding)
    re_encoded_content = decoded_content.encode(to_encoding)
    return json.loads(re_encoded_content.decode(to_encoding))