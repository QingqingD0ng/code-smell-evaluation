class ChoicesValidator:
    def __init__(self, choices):
        self.choices = choices

    def validate_choices_args(self, args):
        return args in self.choices