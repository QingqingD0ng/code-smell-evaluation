import os

def remove_ending_os_sep(input_list):
    if input_list is None:
        return []
    
    def process_string(s):
        return s[:-1] if len(s) > 1 and s[-1] == os.sep else s
    
    return [process_string(item) for item in input_list if isinstance(item, str)]