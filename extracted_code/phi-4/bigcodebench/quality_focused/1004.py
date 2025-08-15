import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt


def task_func(url):
    with urllib.request.urlopen(url) as response:
        text = response.read().decode('utf-8')
    
    words = re.findall(r'\b\w+\b', text)
    word_count = Counter(words)
    most_common_words = word_count.most_common(10)
    
    words, counts = zip(*most_common_words)
    
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Words')
    plt.xticks(rotation=45)
    
    ax = plt.gca()
    plt.tight_layout()
    plt.show()
    
    return word_count, ax