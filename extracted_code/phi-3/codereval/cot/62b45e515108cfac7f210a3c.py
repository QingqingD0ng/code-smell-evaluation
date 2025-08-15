import os

class OCFLStorageRoot:
    def __init__(self, location):
        self.location = location
        self.root_dir = os.path.join(self.location, "ocfl_root")
        os.makedirs(self.root_dir, exist_ok=True)
        self._initialize_ocfl_root()

    def _initialize_ocfl_root(self):
        # Placeholder for OCFL initialization logic
        # This should include creating necessary directories and files
        # based on the OCFL specification.
        pass

    def initialize(self):
        self._initialize_ocfl_root()

# Example usage:
# ocfl_root = OCFLStorageRoot('/path/to/storage')
# ocfl_root.initialize()