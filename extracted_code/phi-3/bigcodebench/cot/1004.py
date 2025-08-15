import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    with urllib.request.urlopen(url) as response:
        text_content = response.read().decode('utf-8')
    words = re.findall(r'\b\w+\b', text_content.lower())
    word_freq = Counter(words)
    most_common_words = word_freq.most_common(10)
    words, frequencies = zip(*most_common_words)
    fig, ax = plt.subplots()
    ax.bar(words, frequencies)
    ax.set_title('Top 10 Most Frequent Words')
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequency')
    plt.xticks(rotation=45)
    plt.tight_layout()
    return word_freq, ax