def __replace_register(flow_params, register_number, register_value):
    if not isinstance(register_number, int) or not isinstance(register_value, int):
        raise ValueError("register_number and register_value must be integers")
    
    if register_number not in flow_params:
        raise KeyError(f"register_number {register_number} not found in flow_params")
    if register_value not in flow_params:
        raise KeyError(f"register_value {register_value} not found in flow_params")
    
    flow_params[register_number] = flow_params[register_value]
    del flow_params[register_value]
    
    return flow_params