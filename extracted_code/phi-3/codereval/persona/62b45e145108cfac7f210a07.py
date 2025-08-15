def validate(self, inventory, extract_spec_version=False):

    self.errors = []


    if not isinstance(inventory, dict):

        self.errors.append("Inventory must be a dictionary.")

        return False


    expected_version = self.spec_version


    if extract_spec_version:

        for item in inventory.values():

            if 'type' in item and'version' in item:

                item_version = item['version']

                if item_version == expected_version:

                    continue

                else:

                    self.errors.append(f"Item type {item['type']} version {item_version} does not match expected version {expected_version}.")

                    return False

            elif 'type' not in item:

                self.errors.append("Item type missing.")

                return False


    else:

        for item in inventory.values():

            if'version' in item and item['version']!= expected_version:

                self.errors.append(f"Item version {item['version']} does not match expected version {expected_version}.")

                return False


    return True