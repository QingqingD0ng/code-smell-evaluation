class SoftwareQualityExpert:
    def __init__(self):
        self.data = {}  # Initialize an empty dictionary to store data

    def get(self, key, default=None):
        """
        If a key exists in the class, the value corresponding to the key is returned. Otherwise, default is returned.
        """
        return self.data.get(key, default)

# Example usage:
expert = SoftwareQualityExpert()
expert.data = {'bug_count': 10, 'test_coverage': 85}

# Retrieve values
print(expert.get('bug_count'))  # Output: 10
print(expert.get('feature_count'))  # Output: None
print(expert.get('feature_count', 0))  # Output: 0