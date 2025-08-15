class TimeUnitsNormalizer:
    def __init__(self):
        self.unit_mapping = {
           'seconds': 1,
           'minutes': 60,
            'hours': 3600,
            'days': 86400
        }

    def normalized(self, value, unit):
        if unit in self.unit_mapping:
            return value * self.unit_mapping[unit]
        else:
            raise ValueError(f"Unknown time unit: {unit}")

# Example usage:
# normalizer = TimeUnitsNormalizer()
# normalized_seconds = normalizer.normalized(5,'minutes')  # Should return 300