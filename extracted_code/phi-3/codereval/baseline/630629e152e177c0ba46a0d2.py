import requests

def retrieve_and_parse_diaspora_webfinger(handle):
    url = f"https://webfinger.diaspora.org/{handle}/.well-known/webfinger"
    response = requests.get(url)
    response.raise_for_status()
    webfinger_data = response.text
    return webfinger_data