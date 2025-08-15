def _replace_url_args(url, url_args):
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    # Parse the URL into components
    parsed_url = urlparse(url)
    query_params = parse_qs(parsed_url.query)

    # Replace query parameters with the ones from url_args
    query_params.update(url_args)

    # Encode the query parameters and reconstruct the URL
    new_query = urlencode(query_params, doseq=True)
    new_url = urlunparse(parsed_url._replace(query=new_query))

    return new_url