import numpy as np
import matplotlib.pyplot as plt
import re
from collections import Counter

def task_func(mystrings, text):
    if not text:
        raise ValueError("Input text must not be empty.")
    
    # Replace spaces with underscores in the given words
    for word in mystrings:
        text = re.sub(re.escape(word), word.replace(" ", "_"), text, flags=re.IGNORECASE)
    
    # Convert text to lowercase and split into words
    words = re.findall(r'\b\w+\b', text.lower())
    
    # Count frequency of each unique word
    word_counts = Counter(words)
    
    # Extract unique words in the order they first appear
    unique_words = []
    seen = set()
    for word in words:
        if word not in seen:
            seen.add(word)
            unique_words.append(word)
    
    # Get frequencies in the order of unique words
    frequencies = [word_counts[word] for word in unique_words]
    
    # Plotting
    fig, ax = plt.subplots()
    ax.bar(unique_words, frequencies)
    
    return ax