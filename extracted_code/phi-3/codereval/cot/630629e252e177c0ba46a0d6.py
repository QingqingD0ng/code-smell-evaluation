import requests
from xrdpy.core.xrd_instance import XRD

def retrieve_diaspora_host_meta(host):
    url = f"http://{host}/api/meta"
    response = requests.get(url)
    response.raise_for_status()
    meta_data = response.json()
    return XRD(meta_data)