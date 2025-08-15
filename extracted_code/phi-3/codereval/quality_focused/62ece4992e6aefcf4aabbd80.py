import os

def remove_ending_os_sep(input_list):
    if input_list is None:
        return []
    return [item[:-1] if len(item) > 1 and item[-1] == os.sep else item for item in input_list]