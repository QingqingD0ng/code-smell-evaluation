def status_str(self, prefix=''):
    return ''.join(f"{prefix}{msg}" for msg in sorted(self.messages))