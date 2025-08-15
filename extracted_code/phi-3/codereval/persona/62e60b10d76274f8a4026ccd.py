def data(self, *keys):
    processed_keys = [self.transform(key) for key in keys]
    return processed_keys