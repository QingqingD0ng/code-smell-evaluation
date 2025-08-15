import json
import requests
import chardet

API_URL = "http://api.example.com/data"
TIMEOUT = 5

def fetch_data(url):
    try:
        response = requests.get(url, timeout=TIMEOUT)
        response.raise_for_status()
        return response.content
    except requests.RequestException:
        return None

def detect_encoding(content):
    if not content:
        return None
    detected = chardet.detect(content)
    if not detected['confidence']:
        raise ValueError("Unable to detect encoding for non-empty content")
    return detected['encoding']

def parse_json(content, from_encoding, to_encoding):
    try:
        return json.loads(content.decode(from_encoding).encode(to_encoding).decode(to_encoding))
    except (UnicodeDecodeError, json.JSONDecodeError):
        return {}

def task_func(url=API_URL, from_encoding=None, to_encoding="utf8"):
    content = fetch_data(url)
    if content is None:
        return {}
    
    if from_encoding is None:
        from_encoding = detect_encoding(content)
    
    return parse_json(content, from_encoding, to_encoding)