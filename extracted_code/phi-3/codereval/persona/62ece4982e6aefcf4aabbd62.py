def _replace_url_args(url, url_args):
    from urllib.parse import urlparse, parse_qs, urlencode, urlunparse

    # Parse the URL into components
    parsed_url = urlparse(url)
    # Parse query parameters into a dictionary
    query_params = parse_qs(parsed_url.query)

    # Replace values in query_params with values from url_args
    query_params.update(url_args)
    # Reconstruct the URL with updated query parameters
    updated_url = urlunparse(parsed_url._replace(query=urlencode(query_params, doseq=True)))

    return updated_url