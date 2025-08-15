def update_last_applied_manifest_dict_from_resp(last_applied_manifest, observer_schema, response):
    for key, observer in observer_schema.items():
        if key in response:
            last_applied_manifest[key] = response[key]
        elif observer.get('default'):
            last_applied_manifest[key] = observer['default']
        elif observer.get('update_fn'):
            last_applied_manifest[key] = observer['update_fn'](response)
        elif 'list' in observer:
            update_last_applied_manifest_list_from_resp(
                last_applied_manifest.setdefault(key, []),
                observer['list'],
                response
            )
        else:
            raise KeyError(f"{key} does not exist in the Kubernetes response and no default or update function defined.")