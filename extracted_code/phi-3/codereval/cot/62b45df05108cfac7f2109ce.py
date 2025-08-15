import os

class OCFLValidator:
    def __init__(self):
        self.ocfl_root_pattern = "/ocfl_root"
        self.pyfs_root_pattern = "/pyfs_root"

    def validate(self, path):
        if os.path.isdir(path) and self.ocfl_root_pattern in path:
            return True
        if os.path.isdir(path) and self.pyfs_root_pattern in path:
            return True
        return False

# Usage
validator = OCFLValidator()
result = validator.validate("/path/to/check")