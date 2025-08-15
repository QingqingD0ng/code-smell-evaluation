import os

def _should_attempt_c_optimizations():
    # Check if the environment variable C_OPTIMIZATIONS is set to 'true'
    return os.getenv('C_OPTIMIZATIONS', 'false').lower() == 'true'