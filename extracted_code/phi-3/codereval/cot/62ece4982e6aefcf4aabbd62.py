from urllib.parse import urlparse, urlunparse, parse_qs, urlencode

def _replace_url_args(url, url_args):
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)
    
    for key, value in url_args.items():
        if key in query_params:
            query_params[key] = value
    
    query_string = urlencode(query_params, doseq=True)
    new_url = urlunparse(parsed_url._replace(query=query_string))
    
    return new_url