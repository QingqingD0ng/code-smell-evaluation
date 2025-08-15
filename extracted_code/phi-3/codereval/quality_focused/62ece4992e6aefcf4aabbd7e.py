def _resolve_string(matcher):
    name = matcher['name']
    default = matcher.get('default')
    value = os.getenv(name)
    if value is None and default is None:
        raise EnvironmentError(f"Required environment variable '{name}' is not set.")
    return value if value is not None else default