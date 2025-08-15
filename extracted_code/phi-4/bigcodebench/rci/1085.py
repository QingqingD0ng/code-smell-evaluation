import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(text, top_n=10):
    if not isinstance(text, str):
        raise ValueError("Input text must be a string.")
    
    text = text.lower()
    text = re.sub(r'[^\w\s]', '', text)
    words = text.split()
    if not words:
        return [], None
    
    word_counts = Counter(words)
    common_words = word_counts.most_common(top_n)
    
    words, counts = zip(*common_words)
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_title('Top Common Words')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    
    return common_words, ax

# Example usage:
# common_words, ax = task_func("This is a sample text. This text contains sample words like 'text','sample', and 'words'.")
# plt.show()  # Call plt.show() outside the function to avoid blocking