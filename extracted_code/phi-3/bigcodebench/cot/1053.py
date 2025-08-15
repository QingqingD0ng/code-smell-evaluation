import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        df = pd.read_csv(file_path)
        text_data = df[df.columns[0]] if df.columns[0] == 'Text' else df[0]
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        X = vectorizer.fit_transform(text_data)
        word_counts = X.sum(axis=0).A1
        vocab = vectorizer.get_feature_names_out()
        top_words = sorted(zip(vocab, word_counts), key=lambda x: x[1], reverse=True)[:10]
        words, counts = zip(*top_words)
        plt.bar(words, counts)
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Common Words')
        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()
        return plt.gca() if not save_path else None
    except FileNotFoundError as e:
        print(f"File not found: {e}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None