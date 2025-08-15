import hashlib
import os

class VersionValidator:
    def __init__(self, root_inventory_path):
        self.root_inventory_path = root_inventory_path
        self.digests = {}

    def validate_version_inventories(self, version_dirs):
        for version_dir in version_dirs:
            inventory_path = os.path.join(self.root_inventory_path, version_dir, 'inventory')
            if not os.path.exists(inventory_path):
                raise FileNotFoundError(f"Inventory for version {version_dir} not found.")
            
            with open(inventory_path, 'r') as inventory_file:
                content = inventory_file.read()
                digest = hashlib.sha256(content.encode('utf-8')).hexdigest()
                
                if digest in self.digests:
                    raise ValueError(f"Digest conflict for version {version_dir}: {digest}")
                
                self.digests[digest] = version_dir

# Example usage:
# root_inventory_path = '/path/to/root/inventory'
# version_dirs = ['1.0', '2.0', '3.0']
# validator = VersionValidator(root_inventory_path)
# validator.validate_version_inventories(version_dirs)