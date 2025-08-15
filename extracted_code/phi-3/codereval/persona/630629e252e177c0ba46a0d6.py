import requests


def retrieve_diaspora_host_meta(host):

    url = f"http://{host}/api/v2/host_meta.json"

    response = requests.get(url)

    if response.status_code == 200:

        return response.json()

    else:

        return None