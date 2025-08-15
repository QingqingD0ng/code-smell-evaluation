class AssetManager:

    def __init__(self):

        self._assets = {}


    def add_asset(self, basename, file_path):

        self._assets[basename] = file_path