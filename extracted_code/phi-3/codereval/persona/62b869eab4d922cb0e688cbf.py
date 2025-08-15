import yaml

from kubernetes import client, config


def generate_default_observer_schema(app):

    config.load_kube_config()


    # Assuming 'app' has a'spec.manifest' which is a list of resource specifications

    for resource_spec in app['spec']['manifest']:

        resource_type = resource_spec['type']

        if not resource_spec.get('observer_schema'):

            resource_spec['observer_schema'] = get_default_observer_schema(resource_type)


    return app


def get_default_observer_schema(resource_type):

    # Define default observer schema for different Kubernetes resource types

    default_schemas = {

        'Pod': {

            'type': 'object',

            'properties': {

               'metadata': {

                    'type': 'object',

                    'properties': {

                        'uid': {'type':'string'},

                        'name': {'type':'string'},

                        'labels': {'type': 'object'},

                        'annotations': {'type': 'object'},

                    },

                },

               'spec': {

                    'type': 'object',

                    'properties': {

                        'containers': {'type': 'array'},

                        'volumes': {'type': 'array'},

                       'restartPolicy': {'type':'string'},

                        'terminationGracePeriodSeconds': {'type': 'integer'},

                    },

                },

               'status': {

                    'type': 'object',

                    'properties': {

                        'phase': {'type':'string'},

                        'podIP': {'type':'string'},

                        'hostIP': {'type':'string'},

                    },

                },

            },

        },

        # Add default schemas for other resource types like Deployment, Service, etc.

    }


    return default_schemas.get(resource_type, {})