import requests
from urllib.parse import urlparse

def get_nodeinfo_well_known_document(url, document_path=None):
    parsed_url = urlparse(url)
    nodeinfo_url = f"{parsed_url.scheme}://{parsed_url.netloc}/nodeinfo"

    try:
        response = requests.get(nodeinfo_url)
        response.raise_for_status()
        node_info = response.json()
        result = {
            "url": nodeinfo_url,
            "document_path": document_path if document_path else node_info.get("documentPath", "None provided"),
            "node_id": node_info.get("nodeId", "None provided"),
            "name": node_info.get("name", "None provided")
        }
        return result
    except requests.RequestException as e:
        return {"error": str(e)}