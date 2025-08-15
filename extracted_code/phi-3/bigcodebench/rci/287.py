import os
import json
from collections import Counter

def read_and_count_words(file_path):
    try:
        with open(file_path, 'r') as f:
            text = f.read()
            words = text.split()
            return Counter(words)
    except Exception as e:
        print(f"Error reading file {file_path}: {e}")
        return Counter()

def task_func(filename, directory):
    word_count = Counter()
    try:
        for file_name in os.listdir(directory):
            if file_name.endswith('.txt'):
                file_path = os.path.join(directory, file_name)
                if os.path.isfile(file_path):
                    file_word_count = read_and_count_words(file_path)
                    word_count.update(file_word_count)
        total_words = sum(word_count.values())
        with open(os.path.join(directory, filename), 'w') as json_file:
            json.dump(word_count, json_file)
    except Exception as e:
        print(f"Error processing directory {directory}: {e}")
        return 0
    return total_words

# Example usage (uncomment and run in a suitable environment):
# if __name__ == "__main__":
#     print(task_func('word_count.json', './testdir/'))