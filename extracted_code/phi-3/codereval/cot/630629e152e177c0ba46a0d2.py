import requests

def retrieve_and_parse_diaspora_webfinger(handle):
    webfinger_url = f"https://diaspora.example.com/webfinger?resource=acct:{handle}@"
    response = requests.get(webfinger_url)
    response.raise_for_status()
    data = response.json()
    return data