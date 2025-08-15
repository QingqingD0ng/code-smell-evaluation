import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(text1, text2):
    text1, text2 = ALPHANUMERIC.sub(' ', text1).lower(), ALPHANUMERIC.sub(' ', text2).lower()
    tokens1, tokens2 = text1.split(), text2.split()
    counter1, counter2 = Counter(tokens1), Counter(tokens2)
    all_tokens = set(tokens1).union(set(tokens2))
    vector1, vector2 = [counter1.get(token, 0) for token in all_tokens], [counter2.get(token, 0) for token in all_tokens]
    dot_product = np.dot(vector1, vector2)
    magnitude1 = np.sqrt(sum([val**2 for val in vector1]))
    magnitude2 = np.sqrt(sum([val**2 for val in vector2]))
    cosine_similarity = dot_product / (magnitude1 * magnitude2) if magnitude1 and magnitude2 else 0
    levenshtein_ratio = ratio(text1, text2)
    return cosine_similarity, levenshtein_ratio