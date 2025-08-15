def _validate_labels(labels):
    for key, value in labels.items():
        validate_key(key)
        validate_value(value)