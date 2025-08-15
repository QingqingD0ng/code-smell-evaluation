import argparse

class CLIExtension:
    def extend_cli(self, root_subparsers):
        parser = argparse.ArgumentParser(description='Main application CLI')
        root_subparsers = parser.add_subparsers(help='sub-command help')

        run_parser = root_subparsers.add_parser('run', help='Execute the application')
        run_parser.add_argument('--port', type=int, help='Port number to run the application on')

        # Add more subparsers for other commands as needed

        return parser