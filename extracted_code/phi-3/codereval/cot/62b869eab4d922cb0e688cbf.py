class KubernetesObserverSchemaGenerator:
    def __init__(self, app):
        self.app = app

    def generate_default_observer_schema(self):
        manifest = self.app.get_manifest()
        for resource_name, resource_info in manifest.items():
            if 'observer_schema' not in resource_info:
                self._generate_default_schema(resource_name, resource_info)

    def _generate_default_schema(self, resource_name, resource_info):
        default_schema = {
            'type': 'object',
            'properties': {
                resource_name: {
                    'type': 'object',
                    'properties': {
                       'status': {
                            'type': 'object',
                            'properties': {
                                'conditions': {
                                    'type': 'array',
                                    'items': {
                                        'type': 'object',
                                        'properties': {
                                            'type': {'type':'string'},
                                           'status': {'type':'string'}
                                        },
                                       'required': ['type','status']
                                    }
                                }
                            },
                            'additionalProperties': False
                        }
                    },
                   'required': ['status', 'conditions']
                }
            }
        }
        resource_info['observer_schema'] = default_schema

# Usage example:
# app = YourKubernetesAppClass()
# generator = KubernetesObserverSchemaGenerator(app)
# generator.generate_default_observer_schema()