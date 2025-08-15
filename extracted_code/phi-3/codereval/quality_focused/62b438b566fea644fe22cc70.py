import argparse
import inspect
import subprocess
import os

def get_parser_arguments(parser):
    return [(action.dest, action.help) for action in parser._actions if isinstance(action, argparse._StoreAction)]

def generate_bash_completion(script_path):
    if not os.path.isfile(script_path):
        return

    with open(script_path, 'r') as script_file:
        script_content = script_file.read()

    # Find the parser
    parser = None
    for line in script_content.split('\n'):
        if line.strip().startswith('parser ='):
            _, parser_str = line.split('=')
            parser = eval(parser_str.strip())
            break

    if parser is None:
        print("Borgmatic script does not contain a parser.")
        return

    # Get parser arguments
    parser_arguments = get_parser_arguments(parser)

    # Generate bash completion script
    bash_completion_script = "\n".join(
        [
            '#!/bin/bash',
            '__borgmatic_completion()',
            '{'
           ' local cur prev opts cur_arg cur_word cur_pos cur_line'
           ' COMPREPLY=()'
           ' cur prev=""'
           ' COMP_LINE="${COMP_LINE%%$'\''*}"'
           ' COMP_POINT="${COMP_POINT}"'
           ' cur_word="${COMP_WORDS[$COMP_CWORD]}"'
           ' cur_pos="${COMP_CWORD}"'
           ' COMP_WORDS=("${COMP_WORDS[*]}")'
           ' COMP_LINE="${COMP_LINE#*?}"'
           ' COMP_WORDS=("${COMP_WORDS[@]/#/}")'
           ' COMP_POINT="${COMP_POINT%%?}"'
           ' cur=""'
           ' while [ -n "${cur_word}" ] && [ -z "${prev}" ]; do'
           '   case "$cur_word"