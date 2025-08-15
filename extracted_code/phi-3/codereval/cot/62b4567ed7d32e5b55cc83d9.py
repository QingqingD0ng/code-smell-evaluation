class MappingNode:
    def __init__(self, value=None, children=None):
        self.value = value
        self.children = children if children is not None else {}

    def merge(self, other):
        for key, child in other.children.items():
            if key in self.children:
                self.children[key].merge(child)
            else:
                self.children[key] = child

def deep_merge_nodes(nodes):
    root = MappingNode()
    for node in nodes:
        root.merge(node)
    return root.children