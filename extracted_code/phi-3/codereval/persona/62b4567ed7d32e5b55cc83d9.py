import collections
import copy

class MappingNode(dict):
    pass

def deep_merge_nodes(nodes):
    merged = MappingNode()

    for node in nodes:
        for key, value in node.items():
            if isinstance(value, MappingNode):
                if key in merged:
                    merged[key] = deep_merge_nodes([merged[key], value])
                else:
                    merged[key] = copy.deepcopy(value)
            else:
                if key in merged:
                    merged_value = merged[key]
                    if isinstance(merged_value, MappingNode):
                        merged[key] = deep_merge_nodes([merged_value, value])
                    else:
                        merged[key] = value
                else:
                    merged[key] = copy.deepcopy(value)
    return merged

# Example usage:
node1 = MappingNode({'a': 1, 'b': {'c': 3}})
node2 = MappingNode({'b': {'d': 4}, 'e': 5})
node3 = MappingNode({'a': MappingNode({'x': 6}), 'f': 7})

merged_nodes = deep_merge_nodes([node1, node2, node3])
print(merged_nodes)