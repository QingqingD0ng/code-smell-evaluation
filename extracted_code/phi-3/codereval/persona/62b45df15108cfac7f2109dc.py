def status_str(self, prefix=''):
    return '\n'.join(f"{prefix}{message}" for message in sorted(self.messages))