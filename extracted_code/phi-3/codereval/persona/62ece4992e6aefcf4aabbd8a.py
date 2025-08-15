from configparser import ConfigParser, NoSectionError, ParsingError

import logging

import os


def load_configurations(config_filenames, overrides=None, resolve_env=True):

    configs = {}

    error_logs = []


    for filename in config_filenames:

        try:

            parser = ConfigParser(inline_comment_prefixes=('#', ';'), resolve_environment=resolve_env)

            with open(filename) as f:

                parser.read_file(f)

            config = parser

            for section in parser.sections():

                config[section] = dict(parser.items(section))

            configs[filename] = config

        except (FileNotFoundError, PermissionError) as e:

            error_logs.append(logging.LogRecord(name='load_configs', level=logging.ERROR, pathname='', lineno=0,

                                                msg=f'Error loading {filename}: {e}', args=None))

        except (NoSectionError, ParsingError) as e:

            error_logs.append(logging.LogRecord(name='load_configs', level=logging.ERROR, pathname='', lineno=0,

                                                msg=f'Error parsing {filename}: {e}', args=None))


    return configs, error_logs