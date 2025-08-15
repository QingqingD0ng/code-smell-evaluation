def update_last_applied_manifest_list_from_resp(last_applied_manifest, observer_schema, response):
    for field in observer_schema:
        if field in response:
            if field not in last_applied_manifest:
                last_applied_manifest.append(response[field])
        elif field =='metadata' and 'name' in response:
            if'metadata' not in last_applied_manifest:
                last_applied_manifest.append(response['metadata'])
        elif field =='metadata' and 'namespace' in response:
            if'metadata' in last_applied_manifest and 'namespace' not in last_applied_manifest['metadata']:
                last_applied_manifest['metadata']['namespace'] = response.get('namespace', None)
        elif field =='metadata' and'selfLink' in response:
            if'selfLink' not in last_applied_manifest:
                last_applied_manifest['selfLink'] = response.get('selfLink', None)
        elif field =='metadata' and 'uid' in response:
            if 'uid' not in last_applied_manifest:
                last_applied_manifest['uid'] = response.get('uid', None)
        elif field =='metadata' and'resourceVersion' in response:
            if'resourceVersion' not in last_applied_manifest:
                last_applied_manifest['resourceVersion'] = response.get('resourceVersion', None)
        elif field =='metadata' and 'generation' in response:
            if 'generation' not in last_applied_manifest:
                last_applied_manifest['generation'] = response.get('generation', None)
        elif field =='metadata' and 'creationTimestamp' in response:
            if 'creationTimestamp' not in last_applied_manifest:
                last_applied_manifest['creationTimestamp'] = response.get('creationTimestamp', None)
        elif field =='metadata' and 'deletionTimestamp' in response:
            if 'deletionTimestamp' not in last_applied_manifest:
                last_applied_manifest['deletionTimestamp'] = response.get('deletion