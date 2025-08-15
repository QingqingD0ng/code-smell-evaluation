class MappingNode(dict):
    pass

def deep_merge_nodes(nodes):
    merged_nodes = MappingNode()
    for node in nodes:
        for key, value in node.items():
            if isinstance(value, dict):
                if key in merged_nodes and isinstance(merged_nodes[key], MappingNode):
                    deep_merge_nodes([merged_nodes[key], value])
                else:
                    merged_nodes[key] = MappingNode(value)
            else:
                merged_nodes[key] = value
    return merged_nodes