def update_last_applied_manifest_list_from_resp(
    last_applied_manifest, observer_schema, response
):

    for obj in response:

        for field in observer_schema:

            if field not in last_applied_manifest:

                last_applied_manifest[field] = obj.get(field, None)