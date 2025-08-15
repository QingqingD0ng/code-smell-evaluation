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
    return dict(root.children)

# Example usage:
node1 = MappingNode()
node1.children["key1"].children = {"subkey1": "value1"}

node2 = MappingNode()
node2.children["key1"].children = {"subkey2": "value2"}

node3 = MappingNode()
node3.children["key2"] = "value3"

merged_nodes = deep_merge_nodes([node1, node2, node3])
print(merged_nodes)