class YourClass:
    def __init__(self):
        self._assets = {}

    def add_asset(self, basename: str, file_path: str) -> None:
        if not basename or not file_path:
            raise ValueError("Basename and file_path must be non-empty strings.")
        self._assets[basename] = file_path

    @property
    def assets(self) -> dict:
        return self._assets
    
    def get_asset(self, basename: str) -> str:
        """Get the file path for the given basename."""
        return self._assets.get(basename, None)