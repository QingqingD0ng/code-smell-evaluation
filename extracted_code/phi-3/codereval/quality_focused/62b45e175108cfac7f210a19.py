class InventoryValidator:
    def validate_fixity(self, fixity, manifest_files):
        # Split the fixity string into lines
        fixity_lines = fixity.strip().split('\n')
        
        # Validate each line in the fixity block
        for line in fixity_lines:
            parts = line.split()
            if len(parts)!= 2:
                return Error()  # Assuming Error is a class that handles errors
            
            file_name, checksum = parts
            
            # Check if the file name is in the manifest
            if file_name not in manifest_files:
                return Error()  # File is not listed in the manifest
            
            # Check if the checksum format is correct (assuming a specific format, e.g., SHA-256)
            if not self._checksum_format(file_name, checksum):
                return Error()  # Checksum format is incorrect

    def _checksum_format(self, file_name, checksum):
        # Implement the checksum format validation logic here
        # For example, assuming a SHA-256 checksum format
        return len(checksum) == 64 and all(c in '0123456789abcdef' for c in checksum.lower())

# Assuming Error is a class that handles errors
class Error(Exception):
    pass