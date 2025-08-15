import collections

class MappingNode(dict):
    pass

def deep_merge_nodes(nodes):
    result = MappingNode()
    keys_seen = set()

    for node in nodes:
        for key, value in node.items():
            if key in result:
                if isinstance(result[key], MappingNode) and isinstance(value, MappingNode):
                    deep_merge_nodes([result[key], value])
                    result[key] = MappingNode(result[key])
                else:
                    result[key] = value
            else:
                result[key] = value
            keys_seen.add(key)

    for key in keys_seen:
        if key not in result:
            del result[key]

    return result