from diaspora.schema import XRD
from diaspora.schema.documents import Document
from diaspora.client import Client
import requests

def retrieve_diaspora_host_meta(host):
    client = Client(host)
    meta_url = client.url + '/meta.xml'
    response = requests.get(meta_url)
    meta_data = Document(response.content)
    return XRD.load(meta_data)