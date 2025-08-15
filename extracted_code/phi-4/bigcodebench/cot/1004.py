import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')
    
    words = re.findall(r'\b\w+\b', text)
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)
    words, counts = zip(*most_common_words)
    
    fig, ax = plt.subplots()
    ax.bar(words, counts)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    ax.set_title('Top 10 Most Frequent Words')
    
    return word_freq, ax

# Example usage (uncomment to run):
# word_freq, ax = task_func('http://www.example.com/data.txt')
# plt.show()