class Status:
    def __init__(self, log):
        self.log = log

    def status_str(self, prefix=''):
        return f"{prefix}{self.log.status_str()}"