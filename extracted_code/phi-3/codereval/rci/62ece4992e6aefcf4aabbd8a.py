import json
import logging
import os
from collections import namedtuple

logging.basicConfig(level=logging.ERROR)
LogRecord = namedtuple('LogRecord', ['levelname', 'filename', 'funcName', 'lineno','msg', 'args', 'exc_info'])

def parse_config_file(filename):
    try:
        with open(filename, 'r') as file:
            return json.load(file), None
    except (FileNotFoundError, PermissionError, json.JSONDecodeError) as e:
        return {}, LogRecord(logging.ERROR, filename, 'parse_config_file', None, str(e))

def merge_configs(configs, overrides):
    for key, value in overrides.items():
        if key in configs and configs[key]!= value:
            raise ValueError(f'Conflict for key {key}: config={configs[key]}, override={value}')
        configs[key] = value
    return configs

def apply_env_variables(configs):
    for key, value in configs.items():
        configs[key] = os.path.expandvars(value)
    return configs

def load_configurations(config_filenames, overrides=None, resolve_env=True):
    configs = {}
    logs = []
    
    for filename in config_filenames:
        config, log_record = parse_config_file(filename)
        if config is not None:
            configs[filename] = config
        else:
            logs.append(log_record)
    
    if overrides:
        configs = merge_configs(configs, overrides)
    
    if resolve_env:
        configs = apply_env_variables(configs)
    
    return configs, logs