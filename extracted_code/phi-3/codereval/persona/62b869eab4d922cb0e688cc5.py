def update_last_applied_manifest_dict_from_resp(last_applied_manifest, observer_schema, response):

    for key, field_schema in observer_schema.items():

        if key not in response:

            raise KeyError(f"Key '{key}' not found in response")

        last_applied_manifest[key] = response[key]