import urllib.request
import re
from collections import Counter
import matplotlib.pyplot as plt

def task_func(url):
    with urllib.request.urlopen(url) as response:
        html_content = response.read().decode('utf-8')
    
    stopwords = set(['the', 'of', 'and', 'to', 'in'])
    words = [word.lower() for word in re.findall(r'\b\w+\b', html_content) if word.lower() not in stopwords]
    
    word_frequencies = Counter(words)
    
    top_words = word_frequencies.most_common(10)
    
    plt.figure()
    plt.bar(*zip(*top_words))
    plt.title('Top 10 Words in Text')
    plt.xlabel('Words')
    plt.ylabel('Frequency')
    plt.show()