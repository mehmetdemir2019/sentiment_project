# src/vectorizer.py
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.pipeline import Pipeline
from sklearn.svm import LinearSVC

def build_text_clf():
    pipe = Pipeline([
        ("tfidf", TfidfVectorizer(
            ngram_range=(1,2),
            min_df=3,
            max_df=0.9,
            sublinear_tf=True
        )),
        ("clf", LinearSVC())
    ])
    return pipe
