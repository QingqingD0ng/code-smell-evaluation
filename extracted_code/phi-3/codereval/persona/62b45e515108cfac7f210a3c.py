import os
import ocfl

class OCFLStorageRootInitializer:
    def __init__(self, path):
        self.path = path
        self.root = None
        self.initialize()

    def initialize(self):
        os.makedirs(self.path, exist_ok=True)
        self.root = ocfl.OCFLStorageRoot(self.path)
        self.root.initialize()

# Usage example:
# initializer = OCFLStorageRootInitializer('/path/to/ocfl_root')