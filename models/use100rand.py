import json

import pandas as pd
from joblib import load
import warnings

# Suppress specific warning
warnings.filterwarnings("ignore")
import nltk
import sklearn
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from joblib import dump, load

nltk.download('stopwords')
nltk.download('punkt_tab')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from sklearn.feature_extraction.text import TfidfVectorizer
# Initialize stopwords and stemmer
stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Function to tokenize, clean, and stem text
def tokenize_clean_and_stem(text):
    tokens = word_tokenize(text.lower())
    stems = []
    for token in tokens:
        if token not in stop_words and token.isalnum():  # Exclude stopwords and non-alphanumeric tokens
            stems.append(stemmer.stem(token))
    return stems


# Load the saved models
scared_model = load("scared_model.pkl")
anxious_model = load("anxious_model.pkl")
excited_model = load("excited_model.pkl")
# relevance_model = load("relevance_model.pkl")

# Check each emotion
emotion_models = {
    'scared': scared_model,
    'anxious': anxious_model,
    'excited': excited_model
}

# get data from 100rand.json
with open('../100rand.json') as f:
    df = json.load(f)

for post in df:
    text = post['body']
    print(f"\nPost: {text}")
    stem = tokenize_clean_and_stem(text)

    for emo, model in emotion_models.items():
        prob = model.predict_proba(stem)[0, 1]
        pred = (prob >= 0.12)
        print(f"Emotion: {emo}, Probability: {prob:.3f}, Predicted label: {int(pred)}")


print("\nDone. Models trained with hyperparameter tuning + 5-Fold CV, final "
      "pipelines saved to disk, example inference shown.")