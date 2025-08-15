class DigestChecker:
    def check_digests_present_and_used(self, manifest_files, digests_used):
        missing_digests = set(self.extract_digests(manifest_files)) - set(digests_used)
        if missing_digests:
            return self.error(f"Missing digests: {', '.join(missing_digests)}")
        return None

    def extract_digests(self, manifest_files):
        digests = set()
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as file:
                for line in file:
                    parts = line.split()
                    if parts and parts[0].startswith('digest'):
                        digests.add(parts[1])
        return digests

    def error(self, message):
        raise Exception(message)