def _c_optimizations_ignored():
    return os.getenv('PURE_PYTHON') not in (None, '0')