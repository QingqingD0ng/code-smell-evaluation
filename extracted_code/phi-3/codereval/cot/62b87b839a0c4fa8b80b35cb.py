class DataProcessor:
    def __init__(self, data):
        self.data = data

    def _get_err_indices(self, coord_name):
        err_indices = [i for i, coord in enumerate(self.data) if coord.get(coord_name, None) is None]
        return err_indices