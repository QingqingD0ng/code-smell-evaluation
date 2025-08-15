from datetime import timedelta


class Structure:

    def __init__(self, days, seconds, microseconds, milliseconds, minutes, hours, weeks):

        self.days = days

        self.seconds = seconds

        self.microseconds = microseconds

        self.milliseconds = milliseconds

        self.minutes = minutes

        self.hours = hours

        self.weeks = weeks


def dehydrate_timedelta(value):

    return Structure(

        days=value.days,

        seconds=value.seconds,

        microseconds=value.microseconds,

        milliseconds=value.seconds % 1 * 1000,

        minutes=value.seconds // 60,

        hours=value.days // 24,

        weeks=value.days // 7

    )