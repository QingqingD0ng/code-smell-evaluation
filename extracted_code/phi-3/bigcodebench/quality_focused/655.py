import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure nltk's stopwords are downloaded
nltk.download('stopwords')

ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = set(nltk.corpus.stopwords.words('english'))

def task_func(texts, num_topics):
    # Preprocessing: remove non-alphanumeric characters, lowercase, remove stopwords
    processed_texts = [ALPHANUMERIC.sub(' ', text.lower()) for text in texts]
    processed_texts = [' '.join(word for word in text.split() if word not in STOPWORDS) for text in processed_texts]
    
    # Vectorization
    vectorizer = TfidfVectorizer()
    tfidf = vectorizer.fit_transform(processed_texts)
    
    # NMF
    model = NMF(n_components=num_topics, random_state=0)
    model.fit(tfidf)
    
    # Extracting the top words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(model.components_):
        top_words = [feature_names[i] for i in topic.argsort()[:-6:-1]]
        topics.append(top_words)
    
    return topics

# Example usage
texts = [
    "Data science involves the study of data.",
    "Machine learning provides systems the ability to learn from data.",
    "Python is a programming language used in data science."
]
topics = task_func(texts, 2)
print(topics)