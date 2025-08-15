def _replace_url_args(url, url_args):
    url_parts = url.split('?', 1)
    if len(url_parts) == 2:
        url_base, url_query = url_parts
        query_parts = url_query.split('&')
        new_query = []
        for part in query_parts:
            key, sep, value = part.partition('=')
            if key in url_args:
                new_query.append(f"{key}={url_args[key]}")
            else:
                new_query.append(part)
        new_url = f"{url_base}?{'&'.join(new_query)}"
    else:
        new_url = url
    return new_url