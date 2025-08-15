class MappingNode(dict):
    pass

def deep_merge_nodes(nodes):
    result = MappingNode()
    for node in nodes:
        for key, value in node.items():
            if key in result:
                if isinstance(result[key], MappingNode) and isinstance(value, MappingNode):
                    deep_merge_nodes([result[key], value])
                else:
                    result[key] = value
            else:
                result[key] = value
    return result