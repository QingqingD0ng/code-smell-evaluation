class ChoiceValidator:
    def __init__(self, valid_choices):
        self.valid_choices = valid_choices

    def validate_choices_args(self, args):
        return args in self.valid_choices

# Example usage:
# validator = ChoiceValidator(['option1', 'option2', 'option3'])
# result = validator.validate_choices_args('option1')  # Returns True
# result = validator.validate_choices_args('option4')  # Returns False