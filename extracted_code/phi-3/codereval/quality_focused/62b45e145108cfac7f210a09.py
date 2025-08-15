class DigestChecker:
    def check_digests_present_and_used(self, manifest_files, digests_used):
        missing_digests = set(self._extract_digests(manifest_files)) - set(digests_used)
        
        if missing_digests:
            return self._raise_error(f"Digests not present: {missing_digests}")
        
        extra_digests = set(digests_used) - set(self._extract_digests(manifest_files))
        if extra_digests:
            return self._raise_error(f"Digests used but not present in manifest: {extra_digests}")

        return None  # or True, depending on the desired return value

    def _extract_digests(self, manifest_files):
        extracted_digests = []
        for manifest in manifest_files:
            with open(manifest, 'r') as file:
                for line in file:
                    if line.startswith('digest'):
                        parts = line.split()
                        extracted_digests.append(parts[1])
        return extracted_digests

    def _raise_error(self, message):
        raise Exception(message)

# Example usage:
# digest_checker = DigestChecker()
# error = digest_checker.check_digests_present_and_used(manifest_files, digests_used)
# if error is not None:
#     print(error)