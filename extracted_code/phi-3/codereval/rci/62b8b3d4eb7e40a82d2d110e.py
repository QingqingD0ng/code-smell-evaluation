import os


def _c_optimizations_ignored():
    pure_python_env = os.getenv('PURE_PYTHON', '0')
    return pure_python_env!= '0'