class RecordExporter:
    def transform(self, key):
        # Placeholder for the transform method implementation
        return key  # Assuming the transform method just returns the key for this example

    def data(self, *keys):
        return [self.transform(key) for key in keys]