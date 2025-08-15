def update_last_applied_manifest_list_from_resp(last_applied_manifest, observer_schema, response):
    # Iterate over each field in the observer_schema
    for field in observer_schema:
        # If the field is not in last_applied_manifest, update it with the corresponding value from response
        if field not in last_applied_manifest:
            last_applied_manifest[field] = response.get(field, None)

    # Iterate over each spec in the'spec' list of the response
    if'spec' in response:
        for spec in response['spec']:
            # If the'spec' contains 'observers', recursively update the last_applied_manifest for the spec
            if 'observers' in spec:
                update_last_applied_manifest_list_from_resp(
                    last_applied_manifest.get(spec['name'], []),
                    observer_schema,
                    spec
                )