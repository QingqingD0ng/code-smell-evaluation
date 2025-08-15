import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    try:
        with urllib.request.urlopen(url) as response:
            text = response.read().decode('utf-8')
    except Exception as e:
        raise RuntimeError(f"Failed to fetch or decode the file: {e}")

    words = re.findall(r'\b\w+\b', text)
    word_counts = Counter(words)
    top_n = 10
    most_common_words = word_counts.most_common(top_n)
    
    words, counts = zip(*most_common_words) if most_common_words else ([], [])
    plt.figure(figsize=(10, 6))
    plt.bar(words, counts)
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.title('Top 10 Most Frequent Words')
    plt.xticks(rotation=45)
    plt.tight_layout()
    ax = plt.gca()
    
    return word_counts, ax