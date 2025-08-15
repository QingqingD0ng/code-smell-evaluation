import hashlib
import os

class VersionValidator:
    def __init__(self):
        self.root_inventory_path = os.path.join("versions", "root", "inventory")
        self.content_digests = {}

    def validate_version_inventories(self, version_dirs):
        self.root_inventory = self._read_inventory_file(self.root_inventory_path)

        for version_dir in version_dirs:
            version_path = os.path.join("versions", version_dir)

            if not os.path.isdir(version_path):
                raise FileNotFoundError(f"Version directory {version_path} does not exist.")

            inventory_path = os.path.join(version_path, "inventory")
            current_inventory = self._read_inventory_file(inventory_path)
            current_hash = hashlib.sha256(current_inventory.encode()).hexdigest()
            self.content_digests[version_dir] = current_hash

            if current_inventory!= self.root_inventory:
                raise ValueError(f"Inventory mismatch in version {version_dir}.")

            self._check_content_digests(version_path)

    def _read_inventory_file(self, file_path):
        with open(file_path, 'r') as inventory_file:
            return inventory_file.read()

    def _check_content_digests(self, version_path):
        for root_path, _, files in os.walk("versions"):
            if root_path!= version_path:
                for file in files:
                    file_path = os.path.join(root_path, file)
                    if file_path not in self.content_digests:
                        content_hash = hashlib.sha256(self._read_file_content(file_path).encode()).hexdigest()
                        self.content_digests[file_path] = content_hash

    def _read_file_content(self, file_path):