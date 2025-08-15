class DataHandler:
    def __init__(self, data):
        self.data = data

    def _get_err_indices(self, coord_name):
        return [i for i, coord in enumerate(self.data) if coord_name in coord]