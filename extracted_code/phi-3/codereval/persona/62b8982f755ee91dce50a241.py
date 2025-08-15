class TimeNormalizer:
    def __init__(self):
        pass

    def seconds_to_integer(self, seconds):
        return int(seconds)

    def minutes_to_integer(self, minutes):
        return int(minutes) * 60

    def hours_to_integer(self, hours):
        return int(hours) * 60 * 60

    def days_to_integer(self, days):
        return int(days) * 24 * 60 * 60

    def weeks_to_integer(self, weeks):
        return int(weeks) * 7 * 24 * 60 * 60

    def normalized(self, time, unit):
        if unit == "seconds":
            return self.seconds_to_integer(time)
        elif unit == "minutes":
            return self.minutes_to_integer(time)
        elif unit == "hours":
            return self.hours_to_integer(time)
        elif unit == "days":
            return self.days_to_integer(time)
        elif unit == "weeks":
            return self.weeks_to_integer(time)
        else:
            raise ValueError("Invalid time unit")