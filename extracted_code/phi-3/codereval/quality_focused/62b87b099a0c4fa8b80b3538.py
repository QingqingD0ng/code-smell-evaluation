class ObjectWithAttributes:
    def __init__(self, fill=None, request=None):
        self.fill = fill
        self.request = request

def is_fill_request_el(obj):
    if hasattr(obj, 'fill') and hasattr(obj,'request'):
        return True
    return False

# Example usage:
# obj_with_attributes = ObjectWithAttributes(fill='data', request='submit')
# print(is_fill_request_el(obj_with_attributes))  # Output: True

# obj_without_attributes = ObjectWithAttributes()
# print(is_fill_request_el(obj_without_attributes))  # Output: False