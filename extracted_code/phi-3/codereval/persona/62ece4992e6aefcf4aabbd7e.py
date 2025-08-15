import os
import re

class EnvironmentVariableError(Exception):
    pass

def _resolve_string(matcher):
    name = matcher['name']
    default = matcher.get('default')
    value = os.getenv(name)
    
    if value is None:
        if default is None:
            raise EnvironmentVariableError(f"Environment variable '{name}' not found and no default value provided.")
        return default
    
    return value

# Example usage:
# Assuming the environment variables 'API_KEY' and 'DB_PASSWORD' are set.
matcher = {'name': 'API_KEY', 'default': 'default_api_key'}
print(_resolve_string(matcher))

# Assuming 'DB_PASSWORD' is not set and no default value provided.
matcher = {'name': 'DB_PASSWORD'}
try:
    print(_resolve_string(matcher))
except EnvironmentVariableError as e:
    print(e)

# Assuming 'DB_PASSWORD' is not set but a default value is provided.
matcher = {'name': 'DB_PASSWORD', 'default':'secure_password'}
print(_resolve_string(matcher))