def generate_default_observer_schema_dict(manifest_dict, first_level=False):
    observer_schema_dict = {}

    for key, value in manifest_dict.items():
        if first_level and isinstance(value, dict):
            continue

        if isinstance(value, dict):
            observer_schema_dict[key] = generate_default_observer_schema_dict(value, first_level)
        elif isinstance(value, list):
            observer_schema_dict[key] = [generate_default_observer_schema_dict(item, first_level) if isinstance(item, dict) else item for item in value]
        else:
            observer_schema_dict[key] = value

    return observer_schema_dict