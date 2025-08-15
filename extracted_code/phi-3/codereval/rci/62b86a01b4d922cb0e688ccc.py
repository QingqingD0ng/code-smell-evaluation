import json

def generate_default_observer_schema_dict(manifest_dict: Dict[str, Any], first_level: bool = False) -> Dict[str, Any]:
    """
    Generate a default observer schema dictionary from a manifest dictionary.
    """
    schema_dict = {}
    for key, value in manifest_dict.items():
        if first_level and isinstance(value, (dict, list)):
            schema_dict[key] = generate_default_observer_schema_dict(value, True)
        elif isinstance(value, dict):
            schema_dict[key] = {k: generate_default_observer_schema_dict(v, first_level) if isinstance(v, (dict, list)) else v for k, v in value.items()}
        elif isinstance(value, list):
            schema_dict[key] = [generate_default_observer_schema_dict(item, first_level) if isinstance(item, (dict, list)) else item for item in value]
        else:
            schema_dict[key] = value
    return schema_dict