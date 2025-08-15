import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

nltk.download('stopwords')
ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = nltk.corpus.stopwords.words('english')

def task_func(texts, num_topics):
    # Preprocess texts
    processed_texts = []
    for text in texts:
        text = ALPHANUMERIC.sub(' ', text).lower()
        words = text.split()
        words = [word for word in words if word not in STOPWORDS]
        processed_texts.append(' '.join(words))
    
    # Vectorize texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Apply NMF to extract topics
    nmf_model = NMF(n_components=num_topics, random_state=1)
    nmf_model.fit(tfidf_matrix)
    
    # Extract topics
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(nmf_model.components_):
        top_features_ind = topic.argsort()[:-11:-1]
        top_features = [feature_names[i] for i in top_features_ind]
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