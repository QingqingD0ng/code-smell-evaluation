import json
import requests
import chardet

API_URL = "http://api.example.com/data"

def task_func(url=API_URL, from_encoding=None, to_encoding="utf8"):
    response = requests.get(url, timeout=5)
    content_bytes = response.content
    
    if not content_bytes:
        return {}
    
    detected_encoding = chardet.detect(content_bytes)['encoding']
    
    try:
        if detected_encoding and not from_encoding:
            content_str = content_bytes.decode(detected_encoding)
        elif from_encoding:
            content_str = content_bytes.decode(from_encoding)
        else:
            raise ValueError("Unable to detect encoding for non-empty content")
        
        return json.loads(content_str.encode(to_encoding))
    except Exception as e:
        print(f"Error: {e}")
        return {}

# Example usage:
# data = task_func()
# print(data)