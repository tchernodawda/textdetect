# AI vs Human Text - Détection de textes générés par l’IA

Une application **Django** utilisant **NLTK**, **spaCy**, **scikit-learn** et **joblib** pour détecter si un texte est écrit par un humain ou généré par une intelligence artificielle.

## Description

Le projet a pour objectif de répondre à un enjeu actuel : **distinguer les textes générés par l’IA de ceux écrits par des humains**.
Nous avons commencé par une introduction sur l’importance de cette tâche, puis nous avons conçu et entraîné un modèle de machine learning, intégré dans une application web permettant aux utilisateurs de tester leurs propres textes.

## Fonctionnalités

* Détection automatique : prédire si un texte est humain ou généré par l’IA.
* Prétraitement du texte : tokenisation, nettoyage et vectorisation TF-IDF.
* Modèle ML : régression logistique avec validation croisée.
* Interface web : application Django permettant de saisir un texte et d’obtenir la prédiction en temps réel.
* Persistance du modèle avec **joblib** pour un rechargement rapide.

## Pipeline du projet

1. **Préparation des données**

   * Extraction des caractéristiques textuelles.
   * Vectorisation TF-IDF pour représenter les textes sous forme numérique.
   * Division en ensembles **entraînement**, **validation** et **test**.

2. **Entraînement du modèle**

   * Utilisation d’une **régression logistique**.
   * Validation croisée pour évaluer la robustesse du modèle.

3. **Évaluation**

   * Test sur des données inédites non utilisées dans l’entraînement.
   * Résultats satisfaisants montrant une bonne capacité de généralisation.

4. **Déploiement**

   * Intégration du modèle dans une application **Django**.
   * Interface web simple pour interagir avec le modèle.

## Technologies utilisées

* **Backend** : Django (Python)
* **Machine Learning** : scikit-learn, joblib
* **NLP** : NLTK, spaCy
* **Vectorisation** : TF-IDF
* **Frontend** : Django templates (HTML, CSS, JS)

## Installation

### Prérequis

* Python 3.8+
* pip
* virtualenv (recommandé)

### Étapes

1. **Cloner le projet**

```bash
git clone https://github.com/thiernodaoudaly/ai-vs-human-text.git
cd ai-vs-human-text
```

2. **Créer et activer un environnement virtuel**

```bash
python -m venv venv
source venv/bin/activate   # Linux/Mac
venv\Scripts\activate      # Windows
```

3. **Installer les dépendances**

```bash
pip install -r requirements.txt
```

4. **Télécharger les ressources NLP**

```python
import nltk
nltk.download('punkt')
nltk.download('stopwords')

python -m spacy download en_core_web_sm
```

5. **Lancer le serveur Django**
   Assurez-vous que le modèle est déjà entraîné et sauvegardé via `joblib`.

```bash
python manage.py runserver
```

6. **Accéder à l’application**
   Ouvrir [http://127.0.0.1:8000](http://127.0.0.1:8000) dans le navigateur.

## Résultats

* Le modèle de régression logistique a montré une bonne précision et une capacité de généralisation.
* La majorité des textes inédits ont été correctement classés comme **Humain** ou **IA**.

## Contribution

Les contributions sont les bienvenues !

1. Fork le repo
2. Crée une branche (`feature/ma-fonctionnalite`)
3. Commit (`git commit -m "Ajout nouvelle fonctionnalité"`)
4. Push (`git push origin feature/ma-fonctionnalite`)
5. Ouvre une Pull Request

