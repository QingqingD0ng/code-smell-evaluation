import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        # Read the CSV file
        df = pd.read_csv(file_path, header=None, names=['Text'] if 'Text' not in pd.read_csv(file_path, nrows=0).columns else None)
        
        # Initialize CountVectorizer with the stop words
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        
        # Fit and transform the text data into word counts
        word_counts = vectorizer.fit_transform(df['Text'])
        
        # Sum up the counts of each word
        sum_words = word_counts.sum(axis=0)
        words_freq = [(word, sum_words[0, idx]) for word, idx in vectorizer.vocabulary_.items()]
        
        # Sort the words by frequency in descending order
        words_freq = sorted(words_freq, key=lambda x: x[1], reverse=True)
        
        # Get the top 10 most common words
        top_words = words_freq[:10]
        top_words_df = pd.DataFrame(top_words, columns=['Word', 'Frequency'])
        
        # Plot the histogram
        plt.figure(figsize=(10, 6))
        plt.bar(top_words_df['Word'], top_words_df['Frequency'], color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Common Words')
        
        # Save or show the plot
        if save_path:
            plt.savefig(save_path)
            plt.close()
        else:
            return plt.gca()
    except FileNotFoundError:
        raise FileNotFoundError(f"File not found: {file_path}")
    except Exception as e:
        print(f"An error occurred: {e}")
        return None