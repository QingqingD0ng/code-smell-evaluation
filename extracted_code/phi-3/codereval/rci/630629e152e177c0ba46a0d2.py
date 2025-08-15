import requests

from bs4 import BeautifulSoup

import re


def retrieve_and_parse_diaspora_webfinger(handle):

    if not isinstance(handle, str) or not handle.strip():

        raise ValueError("Invalid handle provided.")


    url = f"https://webfinger.diaspora.org/{handle}/.well-known/webfinger"


    try:

        response = requests.get(url, timeout=10)

        response.raise_for_status()

    except requests.exceptions.HTTPError as errh:

        print("Http Error:", errh)

    except requests.exceptions.ConnectionError as errc:

        print("Error Connecting:", errc)

    except requests.exceptions.Timeout as errt:

        print("Timeout Error:", errt)

    except requests.exceptions.RequestException as err:

        print("Oops: Something Else", err)


    soup = BeautifulSoup(response.text, 'html.parser')

    identifier_elem = soup.find('a', {'rel':'rel'})

    if identifier_elem:

        identifier = (identifier_elem.get('href').split('?')[0].split('/')[-1]

                      if '/' in identifier_elem.get('href') else identifier_elem.get('id'))

    else:

        raise ValueError("Identifier element not found in webfinger data.")


    name = identifier_elem.text.strip() if identifier_elem else None

    display_name = (soup.find('a', {'rel':'rel'}).find('span', {'class': 'name'}).text.strip()

                    if identifier_elem and identifier_elem.find('span', {'class': 'name'}) else None)


    image_elem = soup.find('link', {'rel': 'enclosure'})

    image = image_elem['href'] if image_elem else None


    return {

        'identifier': identifier,

        'name': name,

        'display