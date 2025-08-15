class TimeNormalizer:
    def __init__(self, time_units):
        self.time_units = time_units

    def normalized(self):
        units_with_values = {'seconds': 1,'minutes': 60, 'hours': 3600, 'days': 86400}
        normalized_values = [unit.lower().replace(' ', '') for unit in self.time_units]
        return [units_with_values[unit] for unit in normalized_values if unit in units_with_values]

# Example usage:
time_units = ['2 hours', '15 minutes', '5 days', '30 seconds']
normalizer = TimeNormalizer(time_units)
print(normalizer.normalized())