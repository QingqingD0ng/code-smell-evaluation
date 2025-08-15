import os
from typing import Optional

class OCFLStorageRoot:
    def __init__(self, path: str):
        self.path: str = path
        self.initialize()

    def initialize(self):
        try:
            if not os.path.exists(self.path):
                os.makedirs(self.path)
        except OSError as e:
            raise Exception(f"An error occurred while creating the directory: {e}")

ocfl_storage_root = OCFLStorageRoot('/path/to/ocfl/storage')