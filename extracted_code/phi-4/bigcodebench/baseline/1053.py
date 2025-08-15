import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
import matplotlib.pyplot as plt

STOP_WORDS = ["a", "an", "the", "in", "on", "at", "and", "or"]

def task_func(file_path, save_path=None):
    try:
        # Load the CSV file
        data = pd.read_csv(file_path, header=None if 'Text' not in pd.read_csv(file_path, nrows=0).columns else 0)
        
        # Ensure the data has a 'Text' column
        if 'Text' not in data.columns:
            data = data.iloc[:, 0]
            data.name = 'Text'
        
        # Initialize CountVectorizer with stop words
        vectorizer = CountVectorizer(stop_words=STOP_WORDS)
        
        # Fit and transform the text data
        word_counts = vectorizer.fit_transform(data['Text'])
        
        # Sum word counts across all documents
        sum_word_counts = word_counts.sum(axis=0)
        
        # Convert to array and get word counts
        word_counts_array = sum_word_counts.A1
        
        # Get feature names to map to word counts
        words = vectorizer.get_feature_names_out()
        
        # Create a DataFrame from word counts
        word_df = pd.DataFrame({'word': words, 'count': word_counts_array})
        
        # Sort by count and get the top 10 words
        top_words = word_df.sort_values(by='count', ascending=False).head(10)
        
        # Plotting
        plt.figure(figsize=(10, 6))
        plt.bar(top_words['word'], top_words['count'], color='skyblue')
        plt.xlabel('Words')
        plt.ylabel('Frequency')
        plt.title('Top 10 Most Common Words')
        
        if save_path:
            plt.savefig(save_path)
            plt.close()
            return None
        else:
            return plt.gca()
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None