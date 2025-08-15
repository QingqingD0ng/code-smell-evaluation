def string_to_int(string, alphabet):
    alphabet_index = {char: index for index, char in enumerate(alphabet)}
    result = 0
    for char in string:
        result = result * len(alphabet) + alphabet_index[char]
    return result