import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')

ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = nltk.corpus.stopwords.words('english')

def preprocess_text(text):
    text = text.lower()
    text = ALPHANUMERIC.sub(' ', text)
    tokens = nltk.word_tokenize(text)
    filtered_tokens = [token for token in tokens if token not in STOPWORDS]
    return''.join(filtered_tokens)

def task_func(texts, num_topics):
    processed_texts = [preprocess_text(text) for text in texts]
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
    "Python is a programming language used in data science."
]
topics = task_func(texts, 2)
print(topics)