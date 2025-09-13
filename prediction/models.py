from django.db import models

import pandas as pd
import joblib
import numpy as np
import string
import spacy
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords, wordnet



stop_words = set(stopwords.words('english'))
linking_words = {'to', 'the', 'and', 'of', 'in', 'on', 'for', 'with', 'at', 'a', 'an'}


# compter le nombre de pnctuations
def count_punctuation_marks(text):
    punctuation_count = sum(1 for char in text if char in string.punctuation)
    return punctuation_count


# compter le nombre de mots de liaison
def count_linking_words(text):
    linking_words_count = sum(1 for word in word_tokenize(
        text.lower()) if word in linking_words)
    return linking_words_count


# Function to calculate percentage of synonyms in a text
def calculate_punctuation_count(text):
    tokens = word_tokenize(text)
    if len(tokens) > 0:
        return count_punctuation_marks(text) / len(tokens) * 100
    else:
        # pour eviter le zerodivisionerror
        return 0

# Function to calculate percentage of synonyms in a text


def calculate_linking_words_count(text):
    tokens = word_tokenize(text)
    if len(tokens) > 0:
        return count_linking_words(text) / len(tokens) * 100
    else:
        return 0

# Fonction pour effectuer l'analyse lexicale et syntaxique d'un texte et calculer le pourcentage de fiabilité lexicale


def analyze_text_and_compute_lexical_reliability(text):
    nlp = spacy.load('en_core_web_sm')
    
    # Traiter le texte avec spaCy
    doc = nlp(text)

    # Initialiser le compteur pour les tokens corrects
    correct_tokens = 0

    # Parcourir chaque token dans le texte
    for token in doc:
        # Vérifier si le token est correct (non-stopword et non-punctuation)
        if not token.is_stop and not token.is_punct:
            # Incrémenter le compteur des tokens corrects
            correct_tokens += 1

    # Calculer le pourcentage de fiabilité lexicale
    lexical_reliability_percentage = (correct_tokens / len(doc)) * 100

    return lexical_reliability_percentage


def preprocess_text_for_prediction(text):
    # Analyse lexicale et syntaxique du texte avec spaCy pour calculer la fiabilité lexicale
    lexical_reliability_percentage = analyze_text_and_compute_lexical_reliability(
        text)

    # Compter le nombre de ponctuations et de mots de liaison
    punctuation_percentage = calculate_punctuation_count(text)
    linking_words_percentage = calculate_linking_words_count(text)

    # Vectorisation du texte avec TfidfVectorizer
    text_vectorized = vectorizer2.transform([text])

    # Concaténation de toutes les fonctionnalités
    features = np.array([
        [punctuation_percentage, linking_words_percentage,
            lexical_reliability_percentage]
    ])

    # Concaténer les vecteurs de texte avec les autres fonctionnalités
    features = np.concatenate((features, text_vectorized.toarray()), axis=1)

    # Retourne les données prêtes pour la prédiction
    return features


df_copy = pd.read_csv(r"final_dataframef21.csv", encoding="utf-8")

# Charger le modèle et le vectorizer
loaded_model2 = joblib.load(r"logistic_regression_model2.pkl")

# Charger le vectoriseur à partir du fichier
vectorizer2 = joblib.load(r"tfidf_vectorizer.pkl")

# # Prétraitement du texte pour la prédiction
# text_to_predict = "Votre texte ici"
# preprocessed_text = preprocess_text_for_prediction(text_to_predict)

# # Prédiction
# prediction = loaded_model2.predict(preprocessed_text)
# print("Le texte est prédit comme:", prediction)

