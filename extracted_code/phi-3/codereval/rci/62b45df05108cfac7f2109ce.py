import fs

class OCFLValidator:
    def __init__(self):
        self.expected_ocfl_metadata_file = '/ocfl_metadata.txt'

    def is_pyfs_root(self, fs):
        return fs.istext()

    def contains_ocfl_root(self, fs):
        return '.o2' in fs.listdir('/')

    def has_ocfl_metadata(self, fs):
        return self.expected_ocfl_metadata_file in fs.listdir('/')

    def validate_pyfs_root(self, fs):
        return self.contains_ocfl_root(fs)

    def validate_ocfl_metadata(self, fs):
        try:
            with fs.open(self.expected_ocfl_metadata_file, 'r') as f:
                metadata = f.read()
                return 'ocfl_version' in metadata and 'ocfl_root_uri' in metadata
        except fs.errors.ResourceNotFound:
            return False

    def validate(self, path):
        try:
            root_fs = fs.open_fs(path, create=False)
            if self.is_pyfs_root(root_fs):
                return self.validate_pyfs_root(root_fs)
            return self.validate_ocfl_metadata(root_fs)
        except fs.errors.CreateFailed:
            return False
        except fs.errors.ResourceNotFound:
            return False
        except fs.errors.PermissionError:
            return False
        except Exception:
            return False

# Usage
# validator = OCFLValidator()
# is_valid = validator.validate('/path/to/check')