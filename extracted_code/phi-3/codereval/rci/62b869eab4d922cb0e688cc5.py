def update_manifest_from_response(manifest, schema, response):
    for field, expected_value in schema.items():
        actual_value = response.get(field)
        if actual_value is not None:
            manifest[field] = actual_value
        elif field not in manifest:
            manifest[field] = expected_value