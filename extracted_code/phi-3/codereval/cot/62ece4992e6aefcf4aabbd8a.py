import json
import logging
import os
from typing import Dict, Sequence, Tuple

# Configure logging
logging.basicConfig(level=logging.ERROR)
logger = logging.getLogger(__name__)

def load_configurations(config_filenames: Sequence[str], overrides: Dict[str, any] = None, resolve_env: bool = True) -> Tuple[Dict[str, dict], Sequence[logging.LogRecord]]:
    configurations = {}
    error_log = []

    for filename in config_filenames:
        try:
            # Check if the file exists
            if not os.path.isfile(filename):
                raise FileNotFoundError(f"Configuration file not found: {filename}")

            # Load the configuration file
            with open(filename, 'r') as file:
                config = json.load(file)

            # Apply overrides if any
            if overrides:
                for key, value in overrides.items():
                    config[key] = value

            # Resolve environment variables if required
            if resolve_env:
                config = resolve_env_variables(config)

            # Validate the configuration
            validate_configuration(config)

            configurations[filename] = config

        except (FileNotFoundError, json.JSONDecodeError, PermissionError) as e:
            logger.error(e, exc_info=True)
            error_log.append(logging.makeLogRecord({'message': str(e), 'filename': __file__, 'lineno': 100, 'funcName': 'load_configurations'}))

    return configurations, error_log

def resolve_env_variables(config: dict) -> dict:
    for key, value in config.items():
        if isinstance(value, str):
            config[key] = os.path.expandvars(value)
        elif isinstance(value, dict):
            config[key] = resolve_env_variables(value)
    return config

def validate_configuration(config: dict) -> None:
    # Placeholder for configuration validation logic
    pass

# Example usage:
# configs, errors