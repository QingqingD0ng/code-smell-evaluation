def merge_extra_vars(vars_dict, extra_vars=None):
    if extra_vars is None:
        extra_vars = []
    
    for var, value in extra_vars:
        vars_dict[var] = value
    return vars_dict