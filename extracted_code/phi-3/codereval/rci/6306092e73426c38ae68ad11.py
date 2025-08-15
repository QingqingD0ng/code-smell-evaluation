def merge_extra_vars(vars_dict, extra_vars=None):
    if not isinstance(vars_dict, dict):
        raise ValueError("vars_dict must be a dictionary")
    if extra_vars is not None and not all(isinstance(var, dict) for var in extra_vars):
        raise ValueError("extra_vars must be a list of dictionaries")
    
    if extra_vars is None:
        return vars_dict
    
    merged_dict = vars_dict.copy()  # Create a copy to avoid mutating the input
    
    for var in extra_vars:
        merged_dict.update(var)  # Directly update the copy without overwriting existing keys
    
    return merged_dict