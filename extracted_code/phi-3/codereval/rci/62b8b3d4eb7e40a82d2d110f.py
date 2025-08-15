import os
from typing import bool

def _should_attempt_c_optimizations() -> bool:
    return os.environ.get('C_OPTIMIZATIONS', 'false').lower() == 'true'