def validate_fixity(self, fixity, manifest_files):
    # Check if the fixity block is a dictionary
    if not isinstance(fixity, dict):
        return "Error: Fixity block must be a dictionary."

    # Check if manifest_files is a list
    if not isinstance(manifest_files, list):
        return "Error: Manifest files must be a list."

    # Check if manifest files are non-empty
    if not manifest_files:
        return "Error: Manifest files list cannot be empty."

    # Check if all manifest files exist in the fixity block
    for file in manifest_files:
        if file not in fixity:
            return f"Error: File '{file}' is not referenced in the fixity block."

    # Check if fixity block has the correct structure
    required_keys = ['algorithm', 'hash', 'digest']
    for key in required_keys:
        if key not in fixity:
            return f"Error: Fixity block missing required key '{key}'."

    # Check if all required keys have the correct type
    for key in required_keys:
        if not isinstance(fixity[key], str):
            return f"Error: Fixity block value for '{key}' must be a string."

    # If all checks pass, return None (implicitly)
    return None