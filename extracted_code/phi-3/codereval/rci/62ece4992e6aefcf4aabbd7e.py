import os

class EnvironmentVariableError(Exception):
    pass

def _resolve_string(matcher):
    name = matcher.get('name')
    default = matcher.get('default')
    value = os.getenv(name)
    if value is None and default is None:
        raise EnvironmentVariableError(f"Environment variable '{name}' is not set and no default value provided.")
    return value if value is not None else default

matcher_dict = {
    'name': 'MY_ENV_VAR',
    'default': 'default_value'
}

try:
    resolved_value = _resolve_string(matcher_dict)
    print(resolved_value)
except EnvironmentVariableError as e:
    print(e)