class TimeConverter:

    SECONDS_PER_MINUTE = 60

    SECONDS_PER_HOUR = 3600


    def parse_time(self, time):

        parts = time.split(':')

        if len(parts)!= 3:

            raise ValueError("Time must be in 'HH:MM:SS' format")

        return tuple(map(int, parts))


    def normalize(self, time):

        hours, minutes, seconds = self.parse_time(time)

        total_seconds = hours * self.SECONDS_PER_HOUR + minutes * self.SECONDS_PER_MINUTE + seconds

        return total_seconds