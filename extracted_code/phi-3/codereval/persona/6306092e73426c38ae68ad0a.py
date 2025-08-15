def validate_choices_args(self, args):

    available_choices = ['option1', 'option2', 'option3']  # Replace with actual choices

    for arg in args:

        if arg not in available_choices:

            raise ValueError(f"Argument {arg} is not a valid choice.")