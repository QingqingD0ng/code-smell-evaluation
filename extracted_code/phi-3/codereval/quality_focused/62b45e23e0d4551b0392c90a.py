import os

import hashlib


def validate_version_inventories(self, version_dirs):

    root_inventory_hash = self.get_root_inventory_hash()

    for version_dir in version_dirs:

        version_inventory_hash = self.get_version_inventory_hash(version_dir)

        if version_inventory_hash!= root_inventory_hash:

            self.record_content_digest_mismatch(version_dir, version_inventory_hash)


def get_root_inventory_hash(self):

    with open(self.root_inventory_path, 'rb') as f:

        file_content = f.read()

    return hashlib.sha256(file_content).hexdigest()


def get_version_inventory_hash(self, version_dir):

    version_inventory_path = os.path.join(version_dir, 'inventory')

    with open(version_inventory_path, 'rb') as f:

        file_content = f.read()

    return hashlib.sha256(file_content).hexdigest()


def record_content_digest_mismatch(self, version_dir, content_digest):

    with open(self.mismatch_record_path, 'a') as f:

        f.write(f"{version_dir}: {content_digest}\n")