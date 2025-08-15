class InventoryValidator:
    def __init__(self, spec_version):
        self.spec_version = spec_version

    def validate(self, inventory, extract_spec_version=False):
        if not isinstance(inventory, dict):
            return False
        required_keys = {'type', 'items'}
        if not required_keys.issubset(inventory.keys()):
            return False
        if extract_spec_version:
            inventory_type = inventory.get('type')
            if not isinstance(inventory_type, str):
                return False
            inventory['spec_version'] = self.determine_spec_version_from_type(inventory_type)
        else:
            inventory['spec_version'] = self.spec_version
        # Additional validation logic based on spec_version
        return True

    def determine_spec_version_from_type(self, type_value):
        # Implement logic to determine spec version from type
        return '1.0'  # Example return value

# Example usage:
# validator = InventoryValidator(spec_version='1.0')
# inventory = {'type': 'commodity', 'items': ['item1', 'item2']}
# is_valid = validator.validate(inventory, extract_spec_version=True)