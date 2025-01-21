import warnings
# Suppress common warnings
warnings.filterwarnings("ignore")

import nltk
import sklearn
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedKFold
from joblib import dump, load

# NLTK setup
nltk.download('stopwords')
nltk.download('punkt')

from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

stop_words = set(stopwords.words('english'))
stemmer = PorterStemmer()

# Custom tokenizer
def tokenize_clean_and_stem(text):
    tokens = word_tokenize(text.lower())
    stems = []
    for token in tokens:
        if token not in stop_words and token.isalnum():
            stems.append(stemmer.stem(token))
    return stems


def train_emotion_model_with_gridsearch(df, label_col):
    """
    Trains a binary logistic regression classifier using TF-IDF and
    hyperparameter tuning with GridSearchCV, then fits a final model
    on all data using the best parameters.
    """
    X = df['post_content'].values
    y = df[label_col].values

    # Define pipeline
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(tokenizer=tokenize_clean_and_stem)),
        ('clf',   LogisticRegression(random_state=42, max_iter=1000))
    ])

    # Possible parameter ranges to try
    param_grid = {
        'tfidf__ngram_range': [(1,1), (1,2)],  # Try unigrams or bigrams
        'tfidf__min_df': [1, 2, 5],            # The cutoff for word frequency
        'clf__penalty': ['l2'],               # Could add 'l1' if you use 'solver'='saga'
        'clf__C': [0.01, 0.1, 1, 5, 10, 100],  # Regularization strength
        # 'clf__class_weight': [None, 'balanced'],  # Uncomment if you suspect class imbalance
    }

    # Stratified K-Fold (5-Fold) for more stable performance estimates
    cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    grid_search = GridSearchCV(
        pipeline,
        param_grid,
        cv=cv,
        scoring='accuracy',    # or 'f1', 'precision', etc.
        n_jobs=-1,            # Use all available CPU cores
        verbose=1             # Logs progress
    )

    # Run hyperparameter search
    grid_search.fit(X, y)

    # Print best results
    print(f"Best params for {label_col}:", grid_search.best_params_)
    print(f"Best CV score for {label_col}: {grid_search.best_score_:.3f}")

    # Retrain on ALL data with best parameters
    best_pipeline = grid_search.best_estimator_
    best_pipeline.fit(X, y)

    return best_pipeline


def main():
    # 1. Load data
    df = pd.read_csv("../relevant_posts.csv")
    print("Data preview:\n", df.head(), "\n")
    print("Total records:", len(df))

    # 2. Define your emotion columns
    emotion_cols = ['scared', 'anxious', 'excited']

    # 3. Train a model for each emotion
    emotion_models = {}
    for emo in emotion_cols:
        print(f"\n--- Training (with hyperparameter tuning) model for emotion: {emo} ---")
        model = train_emotion_model_with_gridsearch(df, label_col=emo)
        emotion_models[emo] = model

        # Save the model
        model_filename = f"../models/{emo}_model.pkl"
        dump(model, model_filename)
        print(f"Saved {emo} model to {model_filename}")



    # 4. Example inference
    print("\n=== Example Inference ===")
    new_post = (
        "i have a yr old anatolian shepherd lbs and is getting increasingly "
        "more anxious and panicky when there is even the slightest bit of thunder "
        "i give her gabapentin and trazadone when i know its going to storm i "
        "have a thunder vest that plays music i have a closet and a kennel she "
        "can go into to feel safe when she starts freaking out she will whine "
        "start pacing and even bark at us for i guess attention but im not "
        "really sure what im supposed to do i try not to coddle or pet her "
        "because ive read its rewarding the panic behavior i try to just "
        "pretend everything is normal but honestly its so hard to watch "
        "when i can feel her anxiety and its sad to see her like that "
        "is there anything else you all do for your storm anxious pups that helps"
    )

    stem = tokenize_clean_and_stem(new_post)


    for emo, model in emotion_models.items():
        prob = model.predict_proba(stem)[0, 1]
        pred = (prob >= 0.5)
        print(f"Emotion: {emo}, Probability: {prob:.3f}, Predicted label: {int(pred)}")

    print("\nDone. Models trained with hyperparameter tuning + 5-Fold CV, final "
          "pipelines saved to disk, example inference shown.")


if __name__ == "__main__":
    main()