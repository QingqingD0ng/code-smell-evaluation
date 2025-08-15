import argparse

def parse_arguments(*unparsed_arguments):
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='subparser_name')
    
    subparser = subparsers.add_parser('subparser_name')
    subparser.add_argument('--option', help='An option for the subparser')
    
    args, unknown_args = parser.parse_known_args(unparsed_arguments)
    
    return {
        'global': args,
       'subparser_name': args
    }

parsed_args = parse_arguments(['--option', 'value','subparser_name', 'another_option', 'value'])
print(parsed_args)