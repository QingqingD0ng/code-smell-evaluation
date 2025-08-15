class InvalidChoiceError(ValueError):
    pass

def validate_choices_args(args, choices):
    if not isinstance(choices, list):
        raise TypeError("Choices must be a list.")
    
    for arg_name, arg_value in args.items():
        if arg_value not in choices:
            raise InvalidChoiceError(f"Invalid choice for {arg_name}: {arg_value}. Allowed choices are: {choices}")

    logging.info("All arguments are valid.")
    return True