import re
from datetime import datetime, timedelta, timezone
from dateutil import tz

class ISO8601Parser:
    DATE_FORMATS = [
        "%Y",
        "%Y-%m",
        "%Y%m",
        "%Y-%m-%d",
        "%Y%m%d",
    ]
    WEEK_FORMAT = "%G-W%V-%u"
    ISO_WEEK_FORMAT = "%G-W%V-%u"

    def parse_date(self, date_str):
        for fmt in self.DATE_FORMATS:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue
        raise ValueError(f"Invalid date format for {date_str}")

    def parse_time(self, time_str):
        time_str = time_str.replace(":", "")
        time_str = time_str.replace("T", " ")
        for fmt in ["%H%M%S%f", "%H%M%S", "%H%M", "%H"]:
            try:
                return datetime.strptime(time_str, fmt).time()
            except ValueError:
                continue
        raise ValueError(f"Invalid time format for {time_str}")

    def parse_iso_week(self, week_str):
        match = re.match(self.ISO_WEEK_FORMAT, week_str)
        if not match:
            raise ValueError(f"Invalid ISO week format for {week_str}")
        year, week, _ = match.groups()
        return datetime.strptime(f"{year} {week} 0", "%G %V %u").date()

    def parse_tz(self, tz_str):
        tz_str = tz_str.replace("Z", "+00")
        match = re.match(r"([+-])(\d{2}):?(\d{2})", tz_str)
        if not match:
            raise ValueError(f"Invalid timezone format for {tz_str}")
        sign, hours,