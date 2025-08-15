def generate_default_observer_schema_dict(manifest_dict, first_level=False):
    schema_dict = {}

    for key, value in (manifest_dict.items() if first_level else manifest_dict.items()):
        if isinstance(value, dict):
            schema_dict[key] = generate_default_observer_schema_dict(value, first_level=True)
        elif isinstance(value, list):
            schema_dict[key] = [generate_default_observer_schema_dict(item, first_level=True) if isinstance(item, dict) else item for item in value]
        else:
            schema_dict[key] = value

    return schema_dict