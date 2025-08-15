import rpy2.robjects as robjects

def inline_r_setup(code: str):
    robjects.r(f'options(repr.plot.width={robjects.r(code)}')

# Example usage:
# inline_r_setup('width = 80')