class InventoryValidationException(Exception):
    pass

class SpecificationVersionException(InventoryValidationException):
    pass

class InventoryValidator:
    def __init__(self, min_version_major, min_version_minor):
        self.min_version_major = min_version_major
        self.min_version_minor = min_version_minor
        self.spec_version = None
        self.is_valid = False

    def validate_inventory(self, inventory, extract_spec_version=False):
        self.spec_version = self._determine_spec_version(inventory, extract_spec_version)
        if not self._validate_spec_version(self.spec_version):
            raise SpecificationVersionException(f"Invalid specification version: {self.spec_version}")
        self.is_valid = True

    def _determine_spec_version(self, inventory, extract_spec_version):
        if extract_spec_version and 'type' in inventory:
            version_str = inventory['type'].split('-')[0]
            if '-' in version_str:
                self.spec_version = version_str
            else:
                self.spec_version = 'unknown'
        else:
            self.spec_version = inventory.get('type', 'unknown')
        return self.spec_version

    def _validate_spec_version(self, version):
        if '-' not in version:
            return False
        version_major, version_minor = map(int, version.split('-'))
        return (version_major, version_minor) >= (self.min_version_major, self.min_version_minor) and version_major <= self.max_version_major

    def set_max_version(self, max_version_major):
        self.max_version_major = max_version_major

# Example Usage:
validator = InventoryValidator(min_version_major=1, min_version_minor=0)
validator.set_max_version(3)
inventory = {'type': '2-0'}
valid