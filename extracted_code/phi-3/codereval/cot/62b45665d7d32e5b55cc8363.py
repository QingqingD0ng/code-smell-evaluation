import argparse

def make_parsers():
    parser = argparse.ArgumentParser(description='Main parser')
    subparsers = parser.add_subparsers(dest='command')
    
    run_parser = subparsers.add_parser('run', help='Run the application')
    run_parser.add_argument('--input', type=str, help='Input file')

    stop_parser = subparsers.add_parser('stop', help='Stop the application')
    stop_parser.add_argument('--force', action='store_true', help='Force stop without confirmation')

    return parser, subparsers