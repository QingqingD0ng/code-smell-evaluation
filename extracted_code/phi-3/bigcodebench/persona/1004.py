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
    
    fig, ax = plt.subplots()
    words, counts = zip(*most_common_words)
    ax.bar(words, counts)
    ax.set_xlabel('Words')
    ax.set_ylabel('Frequencies')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()
    
    return word_freq, ax