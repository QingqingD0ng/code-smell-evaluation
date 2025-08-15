def validate_fixity(self, fixity, manifest_files):
    if not isinstance(fixity, dict):
        return self.error("Fixity block must be a dictionary.")

    if not isinstance(manifest_files, list):
        return self.error("Manifest files must be provided as a list.")

    referenced_files = set()
    for file_ref in fixity.get('files', []):
        if not isinstance(file_ref, str):
            return self.error("Each file reference must be a string.")
        referenced_files.add(file_ref)

    for file in manifest_files:
        if file in referenced_files:
            continue
        return self.error(f"File {file} in manifest is referenced in the fixity block but not listed in the manifest.")

    return None