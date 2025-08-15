class GraphContextManager:
    def _update_context(self, context):
        # Assuming self.graph has a method to get the properties of the graph
        graph_properties = self.graph.get_properties()
        
        # Update context with graph properties
        for key, value in graph_properties.items():
            if key == 'error':
                for error_name, error_value in value.items():
                    subcontext = context.get('error', {}).setdefault(error_name, {'index': None})
                    subcontext['index'] = error_value.get('index', None)
            else:
                # Add to value context if it's a non-error property
                context.setdefault('value', {}).setdefault(key, []).append(value)