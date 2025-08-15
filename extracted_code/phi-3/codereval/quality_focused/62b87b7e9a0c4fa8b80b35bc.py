class GraphContextUpdater:
    ERROR_INDICES_KEY = 'index'
    ERROR_KEYS = {'x_low', 'y_low', 'z_low'}

    def _update_context(self, context, graph):
        if not isinstance(context, dict):
            raise ValueError("context must be a dictionary")

        for field in self.ERROR_KEYS:
            error_key = f"{field}_{graph.error[field]}"
            if error_key in context.get('error', {}):
                # Simply append the index to avoid code smell
                context['error'][error_key][self.ERROR_INDICES_KEY].append(graph.error[field])
            else:
                # Initialize the error subcontext with the index
                context['error'][error_key] = {self.ERROR_INDICES_KEY: [graph.error[field]]}

# Example usage:
# updater = GraphContextUpdater()
# updated_context = updater._update_context(context, graph)