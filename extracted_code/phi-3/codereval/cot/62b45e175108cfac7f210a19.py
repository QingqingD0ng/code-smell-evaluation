class FixityValidator:
    def validate_fixity(self, fixity, manifest_files):
        if not isinstance(fixity, dict) or 'files' not in fixity:
            return self.error()
        
        referenced_files = set(fixity['files'])
        manifest_files_set = set(manifest_files)
        
        missing_files = referenced_files - manifest_files_set
        if missing_files:
            return self.error(f"Missing files in manifest: {', '.join(missing_files)}")
        
        return True

    def error(self, message=None):
        raise Exception(message)

# Example usage:
# validator = FixityValidator()
# result = validator.validate_fixity(fixity_block, manifest_list)
# if result is True:
#     print("Fixity block is valid.")
# else:
#     print("Fixity block is invalid.")