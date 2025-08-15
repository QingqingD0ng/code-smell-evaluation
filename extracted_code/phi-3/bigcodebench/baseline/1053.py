import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        df = pd.read_csv(file_path)
        if 'Text' not in df.columns:
            df['Text'] = df.iloc[:, 0]
        text_data = df['Text'].astype(str).values.tolist()
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        word_counts = vectorizer.fit_transform(text_data)
        word_freq = dict(zip(vectorizer.get_feature_names(), word_counts.sum(axis=0).tolist()[0]))
        top_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:10]
        words, counts = zip(*top_words)
        plt.bar(words, counts)
        plt.title('Top 10 Most Common Words')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        if save_path:
            plt.savefig(save_path)
            return None
        else:
            plt.show()
            return plt.gca()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        print(e)
        return None