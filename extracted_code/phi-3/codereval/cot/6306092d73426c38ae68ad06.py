def get_silent_args(self, args):
    silent_args = [arg for arg in args if arg.startswith('--no-')]
    return silent_args