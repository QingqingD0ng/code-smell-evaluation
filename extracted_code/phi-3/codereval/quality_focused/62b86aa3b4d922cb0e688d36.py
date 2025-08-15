def _validate_labels(labels):
    for key, value in labels.items():
        if not validate_key(key):
            raise ValueError(f"Invalid key: {key}")
        if not validate_value(value):
            raise ValueError(f"Invalid value for key {key}: {value}")