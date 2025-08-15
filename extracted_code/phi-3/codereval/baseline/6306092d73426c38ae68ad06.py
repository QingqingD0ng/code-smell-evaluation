class YourClass:
    def get_silent_args(self, args):
        # Placeholder for logic to identify silenced arguments
        # Assuming args is a list of (name, value) tuples
        silenced_args = [name for name, value in args if self._is_silenced(name)]
        return silenced_args

    def _is_silenced(self, name):
        # Placeholder for checking if an argument is silenced
        # This needs to be implemented based on the specific rules for silencing
        return False