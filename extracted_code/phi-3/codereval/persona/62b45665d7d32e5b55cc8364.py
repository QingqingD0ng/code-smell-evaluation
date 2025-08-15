import argparse

def parse_subparser_arguments(unparsed_arguments, subparsers):
    namespace = {}
    remaining_arguments = []
    
    # Iterate over each argument in the unparsed_arguments list
    for arg in unparsed_arguments:
        parsed = False
        
        # Try to parse the argument with each subparser's parser
        for subparser_name, subparser in subparsers.items():
            try:
                subparser.parse_known_args([arg], namespace)
                namespace[subparser_name].arguments.append(arg)
                parsed = True
                break
            except argparse.ArgumentError:
                continue
        
        # If the argument was not parsed, add it to the remaining_arguments list
        if not parsed:
            remaining_arguments.append(arg)
    
    return namespace, remaining_arguments