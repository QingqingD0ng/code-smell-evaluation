def generate_default_observer_schema_dict(manifest_dict, first_level=False):

    schema_dict = {}

    for key, value in manifest_dict.items():

        if isinstance(value, dict):

            schema_dict[key] = generate_default_observer_schema_dict(value, first_level=True)

        elif isinstance(value, list):

            schema_dict[key] = [generate_default_observer_schema_dict(item, first_level=True) if isinstance(item, dict) else item for item in value]

        else:

            schema_dict[key] = value  # Assuming the value is a simple type, not further processed


    if first_level and not any(isinstance(value, dict) for value in manifest_dict.values()):

        schema_dict = {key: str(value) for key, value in schema_dict.items()}  # Convert all values to strings if only simple types


    return schema_dict