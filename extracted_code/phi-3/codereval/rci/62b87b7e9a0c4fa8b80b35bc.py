class GraphContextManager:
    ERROR_NAMES = ('x', 'y', 'z')  # Define error names as constants

    def _update_context(self, context):
        graph_properties = self.graph.get_properties()

        # Helper function for updating the error context
        def update_error_subcontext(error_name, error_value):
            subcontext = context.setdefault('error', {}).setdefault(error_name, {'index': None})
            subcontext['index'] = error_value.get('index', None)

        # Update context with graph properties
        for key, value in graph_properties.items():
            if key == 'error':
                for error_name, error_value in value.items():
                    update_error_subcontext(error_name, error_value)
            else:
                # Add to value context if it's a non-error property
                context.setdefault('value', {}).setdefault(key, []).append(value)