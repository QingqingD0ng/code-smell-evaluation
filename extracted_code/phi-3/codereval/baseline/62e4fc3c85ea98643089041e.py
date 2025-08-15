def _inline_r_setup(code: str) -> str:
    return f"R.options(init='{code}')"