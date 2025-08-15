import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(text1, text2):
    # Clean and preprocess the texts
    text1_clean = ALPHANUMERIC.sub(' ', text1).lower()
    text2_clean = ALPHANUMERIC.sub(' ', text2).lower()
    
    # Tokenize the texts
    tokens1 = text1_clean.split()
    tokens2 = text2_clean.split()
    
    # Calculate term frequency vectors
    counter1 = Counter(tokens1)
    counter2 = Counter(tokens2)
    
    # Create a combined set of terms
    all_terms = set(counter1.keys()).union(set(counter2.keys()))
    
    # Create term frequency vectors
    vec1 = [counter1[term] for term in all_terms]
    vec2 = [counter2[term] for term in all_terms]
    
    # Calculate cosine similarity
    vec1 = np.array(vec1)
    vec2 = np.array(vec2)
    cosine_similarity = np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))
    
    # Calculate Levenshtein ratio
    levenshtein_ratio = ratio(text1_clean, text2_clean)
    
    return cosine_similarity, levenshtein_ratio