import os

class EnvironmentResolutionError(Exception):
    pass

def _resolve_string(matcher):
    name = matcher['name']
    default = matcher.get('default')

    value = os.getenv(name)
    if value is None and default is None:
        raise EnvironmentResolutionError(f"Environment variable '{name}' is not set and no default value is provided.")
    
    return value if value is not None else default