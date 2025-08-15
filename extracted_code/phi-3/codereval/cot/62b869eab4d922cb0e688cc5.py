def update_last_applied_manifest_dict_from_resp(last_applied_manifest, observer_schema, response):
    for key, observer in observer_schema.items():
        if 'fieldPath' in observer:
            field_path = observer['fieldPath']
            if 'fieldPath' in response and response['fieldPath'] == field_path:
                if observer['type'] == 'array':
                    if 'items' in observer:
                        update_last_applied_manifest_list_from_resp(
                            last_applied_manifest[key],
                            observer['items'],
                            response['items']
                        )
                    else:
                        last_applied_manifest[key] = response['items']
                elif observer['type'] == 'object':
                    if 'properties' in observer:
                        update_last_applied_manifest_dict_from_resp(
                            last_applied_manifest[key],
                            observer['properties'],
                            response['properties'][field_path]
                        )
                    else:
                        last_applied_manifest[key] = response['properties'][field_path]
            elif 'fieldPath' not in response:
                last_applied_manifest[key] = observer.get('default', None)
        elif observer['type'] == 'object':
            update_last_applied_manifest_dict_from_resp(
                last_applied_manifest[key],
                observer,
                response
            )

def update_last_applied_manifest_list_from_resp(manifest_list, observer_schema, response):
    for index, observer in enumerate(observer_schema):
        if 'fieldPath' in observer:
            field_path = observer['fieldPath']
            if 'fieldPath' in response:
                if response['fieldPath'] == field_path:
                    if 'items' in observer:
                        manifest_list[index] = update_last_applied_manifest_list_from_resp(
                            observer['items'],
                            observer['items'],
                            response['items']
                        )
                    else:
                        manifest_list