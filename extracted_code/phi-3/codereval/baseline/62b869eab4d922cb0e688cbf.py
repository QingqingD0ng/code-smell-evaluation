import yaml

def generate_default_observer_schema(app):
    manifest = app.get('spec', {}).get('manifest', [])
    for resource in manifest:
        resource_name = resource.get('name')
        resource_schema = resource.get('custom_schema', {})
        if not resource_schema:
            # Assuming a default schema structure for the sake of example
            default_schema = {
                'type': 'object',
                'properties': {
                   'metadata': {
                        'type': 'object',
                        'properties': {
                            'name': {'type':'string'}
                        },
                       'required': ['name']
                    },
                   'status': {
                        'type': 'object',
                        'properties': {
                            'phase': {'type':'string', 'enum': ['Pending', 'Running', 'Succeeded', 'Failed', 'Unknown']}
                        },
                       'required': ['phase']
                    }
                },
               'required': ['metadata','status']
            }
            resource['custom_schema'] = default_schema
    return yaml.dump(manifest)