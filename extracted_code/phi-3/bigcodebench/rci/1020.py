import json
import requests
import chardet

API_URL = "http://api.example.com/data"
TIMEOUT = 5  # Timeout in seconds for the API request

def task_func(url=API_URL, from_encoding=None, to_encoding="utf8"):
    """
    Fetches data from a specified REST API URL and processes it for JSON parsing. The function handles different
    encoding scenarios and ensures compatibility with various data formats.

    Parameters:
    - url (str): The URL of the REST API. Default is 'http://api.example.com/data'.
    - from_encoding (str, optional): The original encoding of the data. If None, encoding is auto-detected. Default is None.
    - to_encoding (str): The target encoding format for the data. Default is 'utf8'.

    Returns:
    - dict: The JSON-parsed data after re-encoding. Returns an empty dictionary if the content is empty or an error occurs.
    """
    response = requests.get(url, timeout=TIMEOUT)
    content_bytes = response.content
    
    if not content_bytes:
        return {}
    
    detected_encoding = chardet.detect(content_bytes)["encoding"]
    
    if detected_encoding and not from_encoding:
        content_str = content_bytes.decode(detected_encoding)
    elif from_encoding:
        content_str = content_bytes.decode(from_encoding)
    else:
        raise ValueError("Unable to detect encoding for non-empty content")
    
    try:
        return json.loads(content_str.encode(to_encoding))
    except json.JSONDecodeError as e:
        raise ValueError(f"JSON parsing error: {e}")
    except Exception as e:
        raise Exception(f"An error occurred: {e}")

# Example usage:
# try:
#     data = task_func()
#     print(data)
# except ValueError as ve:
#     print(f"Error: {ve}")
# except Exception as e:
#     print(f"An unexpected error occurred: {e}")