import json
import logging
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def load_configurations(config_filenames, overrides=None, resolve_env=True):
    configurations = {}
    parse_errors = []
    
    for filename in config_filenames:
        try:
            config_path = Path(filename)
            if not config_path.is_file():
                raise FileNotFoundError(f"Configuration file {filename} not found.")
            
            with config_path.open('r') as file:
                config = json.load(file)
                
                # Apply environment variable resolution if needed
                if resolve_env:
                    config = {k: os.path.expandvars(v) if isinstance(v, str) else v for k, v in config.items()}
                
                # Apply overrides
                if overrides and filename in overrides:
                    config.update(overrides[filename])
                
                configurations[filename] = config
        except (FileNotFoundError, json.JSONDecodeError) as e:
            parse_errors.append(logging.LogRecord(name='config_parser', level=logging.ERROR, pathname='', lineno=0,
                                                msg=f"Failed to parse configuration file {filename}: {e}", args=None))
    
    return configurations, parse_errors

# Example usage:
# configs, errors = load_configurations(['config1.json', 'config2.json'], overrides={'config1.json': {'new_key': 'new_value'}})