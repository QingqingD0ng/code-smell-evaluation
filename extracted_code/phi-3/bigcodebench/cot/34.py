import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def task_func(text):
    url_pattern = re.compile(r'https?://\S+|www\.\S+')
    words = re.findall(r'\w+', text.lower())
    filtered_words = [word for word in words if not url_pattern.match(word)]
    
    if not filtered_words:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    
    wordcloud = WordCloud().generate(' '.join(filtered_words))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wordcloud