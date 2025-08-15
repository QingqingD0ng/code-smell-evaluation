class Graph:
    def __init__(self):
        self.error_indices = []
        self.context = {}

    def __update_context(self, context):
        if "error" not in context:
            context["error"] = {}
        for error_name, index in self.error_indices:
            if error_name not in context["error"]:
                context["error"][error_name] = {"index": index}

    def record_error(self, error_name):
        self.error_indices.append((error_name, len(self.error_indices)))

    def get_context(self):
        return self.context

# Example usage:
# graph = Graph()
# graph.record_error("error_E_low")
# context = graph.get_context()
# print(context)