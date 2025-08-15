import collections

def deep_merge_nodes(nodes):
    if not nodes:
        return {}

    merged = collections.defaultdict(dict)

    for node in nodes:
        for key, value in node.items():
            if isinstance(value, collections.Mapping):
                merged[key].update(deep_merge_nodes([value]))
            else:
                merged[key].update({key: value})

    return dict(merged)