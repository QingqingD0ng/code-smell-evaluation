import re
import numpy as np
from collections import Counter
from Levenshtein import ratio

ALPHANUMERIC = re.compile('[\W_]+')

def task_func(text1, text2):
    def clean_text(text):
        return ALPHANUMERIC.sub(' ', text).lower()

    def compute_cosine_similarity(vec1, vec2):
        intersection = set(vec1.keys()) & set(vec2.keys())
        numerator = sum([vec1[x] * vec2[x] for x in intersection])

        sum1 = sum([vec1[x] ** 2 for x in vec1.keys()])
        sum2 = sum([vec2[x] ** 2 for x in vec2.keys()])
        denominator = np.sqrt(sum1) * np.sqrt(sum2)

        if not denominator:
            return 0.0
        else:
            return float(numerator) / denominator

    cleaned_text1 = clean_text(text1)
    cleaned_text2 = clean_text(text2)

    counter1 = Counter(cleaned_text1.split())
    counter2 = Counter(cleaned_text2.split())

    cosine_similarity = compute_cosine_similarity(counter1, counter2)
    levenshtein_ratio = ratio(text1, text2)

    return cosine_similarity, levenshtein_ratio