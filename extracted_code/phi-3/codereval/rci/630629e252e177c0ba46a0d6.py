import requests

class HostMeta:
    def __init__(self, xml_data):
        self.xml_data = xml_data

class HttpClient:
    @staticmethod
    def get(url):
        response = requests.get(url)
        if response.status_code == 200:
            return response.text
        else:
            raise Exception(f"Failed to retrieve host meta, status code: {response.status_code}")

def retrieve_diaspora_host_meta(host):
    url = f"https://{host}/host_meta.xml"
    xml_data = HttpClient.get(url)
    return HostMeta(xml_data)