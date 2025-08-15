class TimeConverter:
    def normalize(self, time):
        # Assuming time is a string in the format "HH:MM:SS"
        h, m, s = map(int, time.split(':'))
        total_seconds = h * 3600 + m * 60 + s
        return total_seconds