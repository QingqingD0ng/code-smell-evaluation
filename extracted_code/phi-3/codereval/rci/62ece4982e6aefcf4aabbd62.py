import urllib.parse

def _replace_url_args(url, url_args):
    parsed_url = urllib.parse.urlparse(url)
    scheme, netloc, path, params, query, fragment = parsed_url

    query_dict = urllib.parse.parse_qs(query)

    for key, value in url_args.items():
        if key in query_dict:
            query_dict[key] = value

    new_query = urllib.parse.urlencode(query_dict, doseq=True)

    new_url = urllib.parse.urlunparse((scheme, netloc, path, params, new_query, fragment))

    return new_url