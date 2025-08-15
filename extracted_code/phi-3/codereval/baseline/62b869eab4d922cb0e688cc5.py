for field in observer_schema:
        if field in response:
            last_applied_manifest[field] = response[field]
        else:
            if field not in last_applied_manifest:
                last_applied_manifest[field] = observer_schema[field]