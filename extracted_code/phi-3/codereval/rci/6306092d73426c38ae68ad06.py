class ConfigurableApp:
    silenced_argument_names = set()

    def _is_silenced(self, name):
        return name in self.silenced_argument_names

    def set_silenced_argument(self, name):
        self.silenced_argument_names.add(name)

    def get_silent_args(self, args):
        return [name for name, _ in args if self._is_silenced(name)]