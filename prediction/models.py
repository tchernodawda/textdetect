from django.db import models

import os
import joblib
import numpy as np
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
import nltk

# ---------------------------------------------------------------------------
# Ensure required NLTK resources are present (safe to call multiple times)
# ---------------------------------------------------------------------------
for resource, path in [
    ("tokenizers/punkt", "punkt"),
    ("tokenizers/punkt_tab", "punkt_tab"),
    ("corpora/stopwords", "stopwords"),
]:
    try:
        nltk.data.find(resource)
    except LookupError:
        nltk.download(path, quiet=True)

# ---------------------------------------------------------------------------
# Load spaCy model once at module level (not inside the function)
# ---------------------------------------------------------------------------
try:
    nlp = spacy.load("en_core_web_sm")
except OSError:
    raise OSError(
        "spaCy model 'en_core_web_sm' not found. "
        "Run: python -m spacy download en_core_web_sm"
    )

# ---------------------------------------------------------------------------
# Load ML artefacts using absolute paths relative to this file
# ---------------------------------------------------------------------------
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

_model_path = os.path.join(BASE_DIR, "logistic_regression_model.pkl")
_vectorizer_path = os.path.join(BASE_DIR, "tfidf_vectorizer.pkl")

if not os.path.exists(_model_path):
    raise FileNotFoundError(
        f"Model file not found: {_model_path}\n"
        "Make sure 'logistic_regression_model.pkl' is at the project root."
    )
if not os.path.exists(_vectorizer_path):
    raise FileNotFoundError(
        f"Vectorizer file not found: {_vectorizer_path}\n"
        "Make sure 'tfidf_vectorizer.pkl' is at the project root."
    )

loaded_model2 = joblib.load(_model_path)
vectorizer2 = joblib.load(_vectorizer_path)

# ---------------------------------------------------------------------------
# Feature engineering helpers
# ---------------------------------------------------------------------------

stop_words = set(stopwords.words("english"))
linking_words = {"to", "the", "and", "of", "in", "on", "for", "with", "at", "a", "an"}


def count_punctuation_marks(text: str) -> int:
    return sum(1 for char in text if char in string.punctuation)


def count_linking_words(text: str) -> int:
    return sum(1 for word in word_tokenize(text.lower()) if word in linking_words)


def calculate_punctuation_count(text: str) -> float:
    tokens = word_tokenize(text)
    return count_punctuation_marks(text) / len(tokens) * 100 if tokens else 0.0


def calculate_linking_words_count(text: str) -> float:
    tokens = word_tokenize(text)
    return count_linking_words(text) / len(tokens) * 100 if tokens else 0.0


def analyze_text_and_compute_lexical_reliability(text: str) -> float:
    """
    Compute the percentage of content tokens (non-stop, non-punct)
    using the module-level spaCy pipeline loaded once at startup.
    """
    doc = nlp(text)
    if len(doc) == 0:
        return 0.0
    correct_tokens = sum(
        1 for token in doc if not token.is_stop and not token.is_punct
    )
    return (correct_tokens / len(doc)) * 100


def preprocess_text_for_prediction(text: str) -> np.ndarray:
    """
    Build the feature matrix for a single input text:
      - punctuation % 
      - linking words %
      - lexical reliability %
      - TF-IDF vector (concatenated)
    """
    lexical_reliability = analyze_text_and_compute_lexical_reliability(text)
    punctuation_pct = calculate_punctuation_count(text)
    linking_words_pct = calculate_linking_words_count(text)

    text_vectorized = vectorizer2.transform([text])

    handcrafted = np.array([[punctuation_pct, linking_words_pct, lexical_reliability]])
    features = np.concatenate((handcrafted, text_vectorized.toarray()), axis=1)
    return features