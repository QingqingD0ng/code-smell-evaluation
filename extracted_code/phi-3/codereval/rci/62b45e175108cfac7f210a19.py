def validate_fixity(self, fixity, manifest_files):
    # Helper functions
    def is_dict(obj):
        return isinstance(obj, dict)

    def is_list(obj):
        return isinstance(obj, list)

    def file_exists_in_fixity(file, fixity_dict):
        return file in fixity_dict

    def has_required_keys(obj, keys):
        return all(key in obj for key in keys)

    # Validate fixity block and manifest file structure and contents
    errors = []

    # Ensure fixity block is a dictionary
    if not is_dict(fixity):
        errors.append("Error: Fixity block must be a dictionary.")

    # Ensure manifest files is a list
    if not is_list(manifest_files):
        errors.append("Error: Manifest files must be a list.")

    # Ensure manifest list is not empty
    if not manifest_files:
        errors.append("Error: Manifest files list cannot be empty.")

    # Ensure each manifest file exists in the fixity block and has correct structure
    required_keys = ['algorithm', 'hash', 'digest']
    for file in manifest_files:
        if not file_exists_in_fixity(file, fixity):
            errors.append(f"Error: File '{file}' is not referenced in the fixity block.")
        if not has_required_keys(fixity.get(file, {}), required_keys):
            missing_keys = [key for key in required_keys if key not in fixity.get(file, {})]
            errors.append(f"Error: File '{file}' is missing required keys: {', '.join(missing_keys)}.")
        for key in required_keys:
            if not isinstance(fixity.get(file, {}).get(key), str):
                errors.append(f"Error: Fixity block value for '{key}' in file '{file}' must be a string.")
    
    return errors if errors else None