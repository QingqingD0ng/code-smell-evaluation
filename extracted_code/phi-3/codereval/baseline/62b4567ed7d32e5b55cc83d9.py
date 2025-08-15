from collections import defaultdict

class MappingNode:
    def __init__(self):
        self.children = defaultdict(MappingNode)

    def merge(self, other):
        for key, value in other.children.items():
            current = self.children[key]
            if isinstance(current, MappingNode) and isinstance(value, MappingNode):
                current.merge(value)
            else:
                self.children[key] = value

def deep_merge_nodes(nodes):
    root = MappingNode()
    for node in nodes:
        root.merge(node)
    return root.children  # Assuming you want to return the merged children, not the root itself.