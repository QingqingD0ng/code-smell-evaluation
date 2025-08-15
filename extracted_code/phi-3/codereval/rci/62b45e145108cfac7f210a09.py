class CustomError(Exception):
    pass

class CheckDigestsPresentAndUsed:
    def check_digests_present_and_used(self, manifest_files, digests_used):
        for manifest_file in manifest_files:
            with open(manifest_file, 'r') as file:
                for line in file:
                    parts = line.strip().split(' ')
                    if len(parts)!= 3:
                        raise CustomError("Invalid manifest entry.")
                    filename, size, digest = parts
                    if digest not in digests_used:
                        print(f"Warning: Digest {digest} for {filename} not used.")
                    if not self.verify_digest_present(filename, size):
                        raise CustomError(f"Digest {digest} for {filename} is not present.")

    def verify_digest_present(self, filename, size):
        # Placeholder for actual digest verification logic
        # Replace with real verification code
        return True

# Example usage:
# check = CheckDigestsPresentAndUsed()
# manifest_files = ['manifest1.txt','manifest2.txt']
# digests_used = {'d41d8cd98f00b204e9800998ecf8427e', 'd3f3d7f5e7d3f4a55f2a0f592f1f1e5a'}
# check.check_digests_present_and_used(manifest_files, digests_used)