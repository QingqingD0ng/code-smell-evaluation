def merge_extra_vars(vars_dict, extra_vars=None):
    if extra_vars is not None:
        vars_dict.update(extra_vars)
    return vars_dict