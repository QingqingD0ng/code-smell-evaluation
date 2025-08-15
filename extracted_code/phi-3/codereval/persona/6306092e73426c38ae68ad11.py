def merge_extra_vars(vars_dict, extra_vars=None):

    if extra_vars is None:

        extra_vars = []

    vars_dict.update({var: vars_dict.get(var, None) for var in extra_vars})