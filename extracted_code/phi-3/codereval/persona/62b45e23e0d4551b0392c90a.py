import hashlib
import json
import os

class VersionInventoryValidator:
    def __init__(self):
        self.root_inventory_digest = self.calculate_digest(self.root_inventory_path)

    @staticmethod
    def calculate_digest(file_path):
        hasher = hashlib.sha256()
        with open(file_path, 'rb') as f:
            buf = f.read()
            hasher.update(buf)
        return hasher.hexdigest()

    def validate_version_inventories(self, version_dirs):
        root_digest = self.root_inventory_digest
        for i, version_dir in enumerate(version_dirs, start=1):
            version_inventory_path = os.path.join(version_dir, 'inventory.json')
            if not os.path.exists(version_inventory_path):
                raise FileNotFoundError(f"Missing inventory file in version {i}")
            with open(version_inventory_path, 'r') as f:
                inventory = json.load(f)
                inventory_digest = self.calculate_digest(version_inventory_path)
                if inventory_digest!= root_digest:
                    raise ValueError(f"Inconsistent inventory digest in version {i}")

        print("All version inventories are consistent with the root inventory.")

# Usage example (assuming `version_dirs` is a list of directory names in correct version sequence):
# validator = VersionInventoryValidator()
# validator.validate_version_inventories(version_dirs)