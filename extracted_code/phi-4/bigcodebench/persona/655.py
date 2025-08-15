import re
import nltk
from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer

# Ensure nltk's stopwords are downloaded
nltk.download('stopwords')

# Constants
ALPHANUMERIC = re.compile('[\W_]+')
STOPWORDS = nltk.corpus.stopwords.words('english')

def task_func(texts, num_topics):
    # Preprocess texts
    processed_texts = [
       ''.join(
            word.lower() for word in ALPHANUMERIC.split(text) 
            if word.lower() not in STOPWORDS and word!= ''
        )
        for text in texts
    ]
    
    # Vectorize processed texts using TF-IDF
    vectorizer = TfidfVectorizer()
    tfidf_matrix = vectorizer.fit_transform(processed_texts)
    
    # Apply NMF to extract topics
    nmf_model = NMF(n_components=num_topics, random_state=42)
    W = nmf_model.fit_transform(tfidf_matrix)
    H = nmf_model.components_
    
    # Extract words for each topic
    feature_names = vectorizer.get_feature_names_out()
    topics = []
    for topic_idx, topic in enumerate(H):
        top_features = topic.argsort()[:-6:-1]  # Get top 5 words for each topic
        topics.append([feature_names[i] for i in top_features])
    
    return topics