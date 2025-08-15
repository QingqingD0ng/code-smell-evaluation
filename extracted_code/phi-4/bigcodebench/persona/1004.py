import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    # Download the text file
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')

    # Use regular expression to find words
    words = re.findall(r'\b\w+\b', text)

    # Count the frequency of each word
    word_counts = Counter(words)

    # Get the ten most common words
    most_common_words = word_counts.most_common(10)

    # Separate words and their counts
    common_words, counts = zip(*most_common_words)

    # Plotting the bar chart
    fig, ax = plt.subplots()
    ax.bar(common_words, counts)

    # Set labels and title
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Frequent Words')

    return word_counts, ax