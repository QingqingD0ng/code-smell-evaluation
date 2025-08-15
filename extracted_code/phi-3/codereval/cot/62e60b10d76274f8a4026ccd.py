class RecordExporter:
    def transform(self, keys):
        # Assuming transform is a method that processes the keys
        # For the sake of example, this method will just return the keys
        return keys

class DataHandler:
    def __init__(self, exporter):
        self.exporter = exporter

    def data(self, *keys):
        return self.exporter.transform(keys)

# Example usage:
exporter = RecordExporter()
handler = DataHandler(exporter)
processed_keys = handler.data('key1', 'key2', 'key3')
print(processed_keys)