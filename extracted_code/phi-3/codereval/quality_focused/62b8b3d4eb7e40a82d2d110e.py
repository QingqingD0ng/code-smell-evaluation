def _c_optimizations_ignored():
    return os.getenv('PURE_PYTHON', '0') not in ('', '0')