class DataHandler:
    def __init__(self):
        # Assuming self.data is a pandas DataFrame with error information
        self.data = None

    def _get_err_indices(self, coord_name):
        # Assuming coord_name is a column in self.data that indicates errors
        return self.data[self.data[coord_name]].index.tolist()