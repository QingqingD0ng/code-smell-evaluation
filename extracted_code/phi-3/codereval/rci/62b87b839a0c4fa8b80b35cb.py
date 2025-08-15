class DataHandler:
    def __init__(self, data):
        self.data = data

    @staticmethod
    def get_error_indices(data: list, coord_name: str) -> list:
        if coord_name not in data[0]:
            raise ValueError(f"{coord_name} not found in dataset.")
        return [i for i, coord in enumerate(data) if coord_name in coord]