import requests

class XRD:
    def __init__(self, data):
        self.data = data

def retrieve_diaspora_host_meta(host):
    response = requests.get(f"https://{host}/host_meta.xml")
    if response.status_code == 200:
        return XRD(response.text)
    else:
        raise Exception("Failed to retrieve host meta")