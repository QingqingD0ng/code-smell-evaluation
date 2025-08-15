import ctypes
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def _c_optimizations_available():
    """Load C optimization module or return False if not available."""
    try:
        # Load the shared library
        c_optimizations = ctypes.CDLL('./libc_optimizations.so')
        logger.info("C optimization module successfully loaded.")
        return c_optimizations
    except OSError as error:
        # Log error without exposing stack trace to user
        error_message = f"Failed to load C optimization module: {error}"
        logger.error(error_message)
        return False

if __name__ == "__main__":
    module = _c_optimizations_available()
    if module:
        logger.info("C optimization module is available.")
        # Additional checks or usage of loaded module can be added here
    else:
        logger.warning("C optimization module is not available.")