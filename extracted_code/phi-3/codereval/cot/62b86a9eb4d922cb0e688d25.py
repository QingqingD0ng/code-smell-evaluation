import re


def get_resource_name_regexes():
    regexes = {
        'Cluster': r'^clusters/(?P<name>[a-zA-Z0-9-_]+)$',
        'ClusterRole': r'^clusters/(?P<name>[a-zA-Z0-9-_]+)/roles/(?P<role>[a-zA-Z0-9-_]+)$',
        'ClusterRoleBinding': r'^clusters/(?P<name>[a-zA-Z0-9-_]+)/rolebindings/(?P<name>[a-zA-Z0-9-_]+)$',
        'Node': r'^nodes/(?P<name>[a-zA-Z0-9-_]+)$',
        # Add more resource name regexes as needed
    }
    return regexes