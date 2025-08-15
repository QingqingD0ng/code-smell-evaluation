import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    response = urllib.request.urlopen(url)
    text = response.read().decode('utf-8')
    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    most_common_words = word_counts.most_common(10)
    
    words, counts = zip(*most_common_words)
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    ax = plt.gca()
    plt.show()
    
    return word_counts, ax