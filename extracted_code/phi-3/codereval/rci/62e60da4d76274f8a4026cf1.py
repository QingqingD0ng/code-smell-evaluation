class DataHandler:

    def __init__(self, data):

        self.data = data

        self.index = {key: value for key, value in data.items()}


    def get_value(self, key):

        return self.index.get(key)


    def values(self, *keys):

        return [self.get_value(key) for key in keys if key in self.index]


class DataHandlerError(Exception):

    pass


class DataHandler:

    def __init__(self, data):

        self.data = data

        self.index = {key: value for key, value in data.items()}


    def get_value(self, key):

        if key not in self.index:

            raise DataHandlerError(f"Key {key} not found.")

        return self.index[key]


    def values(self, *keys):

        return [self.get_value(key) for key in keys if key in self.index]