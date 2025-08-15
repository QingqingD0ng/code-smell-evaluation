class AssetManager:
    def __init__(self):
        self._assets = {}

    def add_asset(self, basename, file_path):
        self._assets[basename] = file_path

# Example usage:
# asset_manager = AssetManager()
# asset_manager.add_asset('image', '/path/to/image.png')
# print(asset_manager._assets)  # Output: {'image': '/path/to/image.png'}