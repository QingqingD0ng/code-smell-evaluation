def validate_choices_args(self, args):
    if not hasattr(self, 'choices') or not self.choices:
        raise ValueError("Choices not defined for validation.")

    for arg_name, arg_value in args.items():
        if arg_value not in self.choices:
            raise ValueError(f"Invalid choice for {arg_name}: {arg_value}. Allowed choices are: {self.choices}")

    return True