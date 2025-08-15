import os

def remove_ending_os_sep(input_list):
    if input_list is None:
        return []
    return [s[:-1] if len(s) > 1 and s[-1] == os.sep else s for s in input_list]