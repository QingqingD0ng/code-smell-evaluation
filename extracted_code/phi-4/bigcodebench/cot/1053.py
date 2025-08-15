import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

# Constants
STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        df = pd.read_csv(file_path, header=None if 'Text' not in pd.read_csv(file_path, nrows=0).columns else 0)
        if 'Text' not in df.columns:
            df.columns = ['Text']
        text_data = df['Text'].astype(str)
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        word_counts = vectorizer.fit_transform(text_data)
        sum_words = word_counts.sum(axis=0)
        words = sum_words.A1
        indices = sum_words.indices
        word_dict = dict(zip(vectorizer.get_feature_names_out(), words))
        sorted_words = sorted(word_dict.items(), key=lambda item: item[1], reverse=True)
        top_words = sorted_words[:10]
        words, counts = zip(*top_words)
        fig, ax = plt.subplots()
        ax.bar(words, counts)
        ax.set_xlabel('Words')
        ax.set_ylabel('Frequency')
        ax.set_title('Top 10 Most Common Words')
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            plt.show()
            return ax

    except FileNotFoundError:
        print(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")

    return None