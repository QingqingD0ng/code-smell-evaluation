import requests

def send_document(url, data, timeout=10, method="post", *args, **kwargs):
    session = requests.Session()
    if method.lower() == "get":
        response = session.get(url, params=data, timeout=timeout, *args, **kwargs)
    else:
        response = session.post(url, data=data, timeout=timeout, *args, **kwargs)
    return response.text