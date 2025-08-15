from collections import Mapping

def deep_merge_nodes(nodes):
    result = Mapping()
    stack = [result]

    for node in nodes:
        current_dict = stack[-1]
        for key, value in node.items():
            while key in current_dict:
                if isinstance(current_dict[key], Mapping) and isinstance(value, Mapping):
                    # Merge nested dictionaries iteratively
                    if not isinstance(current_dict[key], Mapping):
                        current_dict[key] = Mapping()
                    stack.append(current_dict[key])
                    current_dict = stack[-1]
                current_dict[key] = value
            stack.pop()

    return result