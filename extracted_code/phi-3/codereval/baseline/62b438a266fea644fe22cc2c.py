import argparse

def make_parsers():
    parser = argparse.ArgumentParser(description='Main application parser.')
    
    subparsers = parser.add_subparsers(dest='command', help='Sub-command help')
    
    # Example of adding a subparser for a 'run' command
    run_parser = subparsers.add_parser('run', help='Run the application.')
    run_parser.add_argument('--verbose', action='store_true', help='Increase output verbosity')
    
    # Example of adding a subparser for a 'config' command
    config_parser = subparsers.add_parser('config', help='Manage application configuration.')
    config_parser.add_argument('--update', type=str, help='Update configuration file')
    
    # Add more subparsers as needed...

    return parser, subparsers