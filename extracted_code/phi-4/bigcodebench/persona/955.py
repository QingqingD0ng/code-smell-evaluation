import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty.")

    # Replace spaces with underscores in mystrings and create a regex pattern
    pattern = '|'.join(re.escape(s.replace(' ', '_')).lower() for s in mystrings)
    
    # Replace occurrences in the text with the modified strings
    modified_text = re.sub(pattern, lambda m: m.group(0).replace(' ', '_'), text.lower())

    # Split the modified text into words
    words = re.findall(r'\b\w+\b', modified_text)

    # Count the frequency of each word
    word_counts = Counter(words)

    # Prepare data for plotting
    unique_words = list(word_counts.keys())
    frequencies = list(word_counts.values())

    # Plotting
    fig, ax = plt.subplots()
    ax.bar(unique_words, frequencies)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Word Frequency Plot')
    plt.xticks(rotation=45, ha='right')
    
    return ax