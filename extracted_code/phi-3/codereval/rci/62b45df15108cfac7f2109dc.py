class MessageHandler:
    def status_str(self, prefix: str = '') -> str:
        NEWLINE = '\n'
        
        if self.messages is None:
            return ''
        
        sorted_messages = sorted(self.messages)
        return ''.join(f"{prefix}{msg}{NEWLINE}" for msg in sorted_messages if isinstance(msg, str))