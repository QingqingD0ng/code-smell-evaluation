from collections import defaultdict

class MappingNode(dict):
    pass

def deep_merge_nodes(nodes):
    merged = defaultdict(MappingNode)
    for node in nodes:
        for key, value in node.items():
            if isinstance(value, MappingNode):
                merged[key].update(deep_merge_nodes([value]))
            else:
                merged[key].update(value)
    return dict(merged)