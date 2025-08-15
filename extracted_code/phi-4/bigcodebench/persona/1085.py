import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(text):
    # Convert text to lowercase
    text = text.lower()
    
    # Remove punctuation using regex
    text = re.sub(r'[^\w\s]', '', text)
    
    # Split text into words
    words = text.split()
    
    # Count the frequency of each word
    word_counts = Counter(words)
    
    # Get the 10 most common words
    common_words = word_counts.most_common(10)
    
    # Extract words and their counts for plotting
    words, counts = zip(*common_words)
    
    # Plot the bar chart
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    
    # Set chart title and labels
    ax.set_title('Top 10 Most Common Words')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    
    # Show the plot
    plt.xticks(rotation=45)
    plt.tight_layout()
    
    return common_words, ax