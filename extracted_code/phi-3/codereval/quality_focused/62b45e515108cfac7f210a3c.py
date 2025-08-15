from pyocfl import OcfL


class OcfLStorageRoot:

    def __init__(self, root_path):

        self.root_path = root_path

        self.ocfl_root = OcfL.create_root(root_path)


    def initialize(self):

        OcfL.initialize(self.ocfl_root)

        return self.ocfl_root


# Usage

ocfl_root = OcfLStorageRoot('/path/to/ocfl/root')

ocfl_root.initialize()