import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty.")
    
    # Replace spaces in mystrings with underscores
    modified_strings = [s.lower().replace(' ', '_') for s in mystrings]
    
    # Replace occurrences in text with modified versions
    for original, modified in zip(mystrings, modified_strings):
        text = re.sub(r'\b' + re.escape(original) + r'\b', modified, text, flags=re.IGNORECASE)
    
    # Split text into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count frequency of each word
    word_count = Counter(words)
    
    # Filter words to include only those that were modified
    filtered_words = {word: count for word, count in word_count.items() if word in modified_strings}
    
    # Plotting
    unique_words = list(filtered_words.keys())
    frequencies = list(filtered_words.values())
    
    plt.figure(figsize=(10, 6))
    plt.bar(unique_words, frequencies, color='skyblue')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Frequency of Unique Words')
    plt.xticks(rotation=45, ha='right')
    
    ax = plt.gca()
    plt.tight_layout()
    plt.show()
    
    return ax