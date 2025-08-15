class ArgumentSnipper:
    def get_silent_args(self, args):
        return [arg for arg in args if arg.startswith('--') and not arg.endswith('--')]