import re
from wordcloud import WordCloud
import matplotlib.pyplot as plt

def task_func(text):
    # Remove URLs from the text
    text = re.sub(r'http\S+', '', text)
    
    # Generate a word cloud
    wordcloud = WordCloud(width=800, height=400).generate(text)
    
    # Raise an error if no words are available
    if not wordcloud.words_:
        raise ValueError("No words available to generate a word cloud after removing URLs.")
    
    # Plot the word cloud
    plt.figure(figsize=(10, 5))
    plt.imshow(wordcloud, interpolation='bilinear')
    plt.axis('off')
    plt.show()
    
    return wordcloud