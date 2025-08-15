def update_last_applied_manifest_list_from_resp(last_applied_manifest, observer_schema, response):
    for item in observer_schema:
        if item not in last_applied_manifest:
            last_applied_manifest.append(item)
    for item in response:
        if item['name'] not in last_applied_manifest:
            last_applied_manifest.append(item['name'])