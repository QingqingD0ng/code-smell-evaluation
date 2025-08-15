import argparse

class QualityExpert:
    def __init__(self):
        self.parser = argparse.ArgumentParser(description='Quality Expert Argument Parser')
        self.parser.add_argument('command', help='The command to execute')
        self.add_arguments()

    def add_arguments(self):
        self.parser.add_argument('--option1', help='Description for option1', default='default1')
        self.parser.add_argument('--option2', help='Description for option2', default='default2')
        # Add more arguments as needed

    def get_option_spec(self, command_name, argument_name):
        return self.parser.parse_args([command_name, argument_name]).__dict__[argument_name]

# Example usage:
expert = QualityExpert()
option_spec = expert.get_option_spec('quality_check', 'option1')
print(option_spec)