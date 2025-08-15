class InventoryValidator:
    def __init__(self, spec_version):
        self.spec_version = spec_version

    def validate(self, inventory, extract_spec_version=False):
        if not inventory:
            raise ValueError("The inventory cannot be empty.")

        # Check if 'type' key exists and is valid
        if extract_spec_version and 'type' in inventory:
            type_value = inventory['type']
            if not isinstance(type_value, str):
                raise ValueError("The 'type' value must be a string.")
            # Assume type_value contains a valid version string for simplicity
        else:
            type_value = self.spec_version

        # Add additional validation logic here based on the type_value or self.spec_version
        # For example:
        if type_value not in ['v1', 'v2']:
            raise ValueError("Unsupported specification version.")

        # Return True if all validations pass
        return True


# Example usage:
# validator = InventoryValidator('v1')
# inventory = {'type': 'v1', 'items': ['item1', 'item2']}
# is_valid = validator.validate(inventory, extract_spec_version=True)