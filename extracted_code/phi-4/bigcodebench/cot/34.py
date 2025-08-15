import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def task_func(text):
    text = re.sub(r'http\S+|www\.\S+', '', text)
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    if not wordcloud.words_:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wordcloud