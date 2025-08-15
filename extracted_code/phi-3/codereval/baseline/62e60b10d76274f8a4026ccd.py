record_exporter = RecordExporter()  # Assuming RecordExporter is already defined

def data(self, *keys):
    return [record_exporter.transform(key) for key in keys]