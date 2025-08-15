import requests
from xml.etree import ElementTree

def retrieve_and_parse_diaspora_webfinger(handle):
    webfinger_url = f"https://webfinger.readthedocs.io/api/v1/well-known/webfinger/{handle}.well-known?resource=https://{handle}"
    response = requests.get(webfinger_url)
    response.raise_for_status()
    root = ElementTree.fromstring(response.content)
    return {
        'url': root.find('link').attrib['href'],
        'display_name': root.find('displayName').text,
        'identifier': root.find('identifier[@type="oct"]').text,
        'email': root.find('email').text,
        'image': root.find('image') is not None and root.find('image').attrib['href']
    }

# Example usage:
# handle = 'diaspora_user_handle'
# diaspora_info = retrieve_and_parse_diaspora_webfinger(handle)
# print(diaspora_info)