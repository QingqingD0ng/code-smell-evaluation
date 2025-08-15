from typing import Dict, Any

def generate_default_observer_schema(app: Dict[str, Any]) -> Dict[str, Dict[str, Dict[str, Any]]]:
    default_schema = {}
    manifest = app.get('spec', {}).get('manifest', [])

    for resource in manifest:
        resource_name = resource.get('name')
        if resource_name and 'observers' not in resource:
            default_observers = {
                'k8s': {
                    'type': 'Event',
                    'kind': resource_name.lower(),
                   'source': f"{resource_name.lower()}s"
                }
            }
            default_schema[resource_name] = default_observers

    return default_schema