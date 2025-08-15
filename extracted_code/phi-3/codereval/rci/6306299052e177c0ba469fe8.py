def check_sender_and_entity_handle_match(sender_handle, entity_handle):
    if not isinstance(sender_handle, str) or not isinstance(entity_handle, str):
        raise TypeError("Both sender_handle and entity_handle must be strings")
    return sender_handle == entity_handle