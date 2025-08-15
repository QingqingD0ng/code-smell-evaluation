import requests

def send_document(url, data, timeout=10, method="post", *args, **kwargs):
    try:
        response = requests.request(method, url, data=data, timeout=timeout, *args, **kwargs)
        return response
    except requests.exceptions.RequestException as e:
        return None