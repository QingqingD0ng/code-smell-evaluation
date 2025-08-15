import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))  # Using a set for faster lookup

def preprocess_text(text, stopwords=None):
    if stopwords is None:
        stopwords = STOPWORDS
    text = text.lower()
    text = ALPHANUMERIC.sub(' ', text)
    tokens = [token for token in nltk.word_tokenize(text) if token and token not in stopwords]
    return''.join(tokens)

def task_func(texts, num_topics, custom_stopwords=None):
    if not isinstance(texts, list) or not all(isinstance(text, str) for text in texts):
        raise ValueError("Input 'texts' must be a list of strings.")
    if not isinstance(num_topics, int) or num_topics <= 0:
        raise ValueError("Number of topics must be a positive integer.")

    processed_texts = [preprocess_text(text, stopwords=custom_stopwords) for text in texts]
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    model = NMF(n_components=num_topics, random_state=42).fit(tfidf_matrix)
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_features = [feature_names[i] for i in topic.argsort()[:-2:-1]]
        topics.append(top_features)
    return topics

# Example usage
texts = [
    "Data science involves the study of data.",
    "Machine learning provides systems the ability to learn from data.",
    "Python is