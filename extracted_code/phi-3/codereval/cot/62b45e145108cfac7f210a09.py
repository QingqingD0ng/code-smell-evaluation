class DigestChecker:
    def check_digests_present_and_used(self, manifest_files, digests_used):
        missing_digests = {digest for _, digest in manifest_files} - set(digests_used)
        if missing_digests:
            self.error(f"Missing digests: {', '.join(missing_digests)}")
        else:
            self.info("All digests are present and used.")

    def error(self, message):
        raise Exception(message)

    def info(self, message):
        print(message)