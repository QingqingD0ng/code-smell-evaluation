import re
import matplotlib.pyplot as plt
import numpy as np

def task_func(text, rwidth=0.8):
    # Extract words using a regular expression to match sequences of word characters
    words = re.findall(r'\b\w+\b', text)
    
    # Calculate the length of each word
    word_lengths = [len(word) for word in words]
    
    # Check if there are any word lengths to plot
    if not word_lengths:
        # Create an empty plot if no words are present
        fig, ax = plt.subplots()
        ax.set_title('Word Length Distribution')
        ax.set_xlabel('Word Length')
        ax.set_ylabel('Frequency')
        ax.text(0.5, 0.5, 'No words to display', horizontalalignment='center',
                verticalalignment='center', transform=ax.transAxes)
        return ax

    # Create a histogram of word lengths
    fig, ax = plt.subplots()
    ax.hist(word_lengths, bins=range(1, max(word_lengths) + 2), rwidth=rwidth, edgecolor='black')
    
    # Set histogram labels and title
    ax.set_title('Word Length Distribution')
    ax.set_xlabel('Word Length')
    ax.set_ylabel('Frequency')
    
    # Return the Axes object
    return ax