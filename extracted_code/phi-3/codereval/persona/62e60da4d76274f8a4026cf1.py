def values(self, *keys):

    return [self.data[key] for key in self.index if key in self.data]