from argparse import ArgumentParser, ArgumentParser as Parser

class YourCLIClass:
    def __init__(self):
        self.parser = Parser(prog='your_cli')

    def create_spec_subcommand(self):
        spec_parser = self.parser.add_parser('spec', help='Spec sub-command help')
        spec_parser.add_argument('--option1', help='Option 1 help')
        spec_parser.add_argument('--option2', help='Option 2 help')
        # Define more spec-related options and subcommands here

    def extend_cli(self, root_subparsers):
        subparsers = root_subparsers.add_subparsers(help='sub-command help')
        self.create_spec_subcommand()
        # Extend with more subcommands as needed
        root_subparsers.add_parser(self.parser)

# Usage
if __name__ == "__main__":
    top_level_parser = ArgumentParser(prog='your_cli')
    subparsers = top_level_parser.add_subparsers(help='sub-command help')
    cli = YourCLIClass()
    cli.extend_cli(subparsers)
    args = top_level_parser.parse_args()