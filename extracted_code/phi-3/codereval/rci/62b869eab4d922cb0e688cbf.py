from typing import List, Dict, Any
import yaml

def default_observer_schema() -> Dict[str, Any]:
    return {
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

def generate_default_observer_schema(app: Dict[str, Any]) -> str:
    manifest = app.get('spec', {}).get('manifest', [])
    for resource in manifest:
        name = resource.get('name')
        if 'custom_schema' not in resource:
            resource['custom_schema'] = default_observer_schema()
    return yaml.dump(manifest)