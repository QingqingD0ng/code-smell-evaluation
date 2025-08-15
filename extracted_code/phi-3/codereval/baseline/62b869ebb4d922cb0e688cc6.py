def update_last_applied_manifest_list_from_resp(last_applied_manifest, observer_schema, response):
    for field in observer_schema:
        if field not in last_applied_manifest:
            last_applied_manifest[field] = response.get(field, None)
    for spec in response.get('spec', []):
        if 'observers' in spec:
            update_last_applied_manifest_list_from_resp(last_applied_manifest.get(spec['name'], []), observer_schema, spec)