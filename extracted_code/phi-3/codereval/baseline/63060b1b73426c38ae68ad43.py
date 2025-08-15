from argparse import ArgumentParser

class YourCLIClass:
    def extend_cli(self, root_subparsers):
        # Create the parser for the root command
        parser = ArgumentParser(prog='your_cli')
        
        # Add your sub-commands here
        subparsers = parser.add_subparsers(help='sub-command help')
        
        # Example sub-command'spec'
        spec_parser = subparsers.add_parser('spec', help='Spec sub-command help')
        # Add options to'spec' sub-command here
        spec_parser.add_argument('--option1', help='Option 1 help')
        spec_parser.add_argument('--option2', help='Option 2 help')
        
        # Assign the parser to the root_subparsers
        root_subparsers.add_parser(parser)