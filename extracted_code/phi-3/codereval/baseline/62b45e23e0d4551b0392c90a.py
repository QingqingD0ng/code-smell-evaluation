import hashlib
import os

class VersionValidator:
    def __init__(self):
        self.root_inventory = None
        self.content_digests = {}

    def validate_version_inventories(self, version_dirs):
        for version_dir in version_dirs:
            version_path = os.path.join("versions", version_dir)
            if not os.path.isdir(version_path):
                raise FileNotFoundError(f"Version directory {version_path} does not exist.")

            # Read the root inventory file
            root_inventory_path = os.path.join("versions", "root", "inventory")
            with open(root_inventory_path, 'r') as root_inventory_file:
                self.root_inventory = root_inventory_file.read()

            # Compute the hash of the current version's inventory
            inventory_path = os.path.join(version_path, "inventory")
            with open(inventory_path, 'r') as inventory_file:
                current_inventory = inventory_file.read()
                current_hash = hashlib.sha256(current_inventory.encode()).hexdigest()
                self.content_digests[version_dir] = current_hash
            
            # Validate the inventory
            if current_inventory!= self.root_inventory:
                raise ValueError(f"Inventory mismatch in version {version_dir}.")

            # Check against digests of content that may differ from the root inventory
            for root_path, _, files in os.walk("versions"):
                for file in files:
                    file_path = os.path.join(root_path, file)
                    if file_path!= root_inventory_path:
                        with open(file_path, 'rb') as f:
                            file_content = f.read()
                            file_hash = hashlib.sha256(file_content).hexdigest()
                            if file_path not in self.content_digests:
                                self.content_digests[file_path] =