import warnings
import os

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

from sklearn.metrics import roc_curve, auc, roc_auc_score
import matplotlib.pyplot as plt


# Function to tokenize, clean, and stem text
def tokenize_clean_and_stem(text):
    tokens = word_tokenize(text.lower())
    stems = []
    for token in tokens:
        if token not in stop_words and token.isalnum():  # Exclude stopwords and non-alphanumeric tokens
            stems.append(stemmer.stem(token))
    return stems

def train_and_evaluate_loo(df, label_col, penalty='l2', C=0.9, class_weight=None, min_df=0.005, ngram_range=(1,3), max_features=5000):
    X = df['body'].values
    y = df[label_col].values

    # Define a pipeline: TF-IDF vectorization -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_clean_and_stem,
            ngram_range=params['ngram_range'],
            min_df=params['min_df'],
            max_features=params['max_features']
        )),
        ('clf', LogisticRegression(
            penalty='l2',
            C=params['C'],
            class_weight=params['class_weight'],
            solver='liblinear',
            random_state=42
        ))
    ])

    # Leave-One-Out cross-validation
    loo = LeaveOneOut()
    scores = cross_val_score(pipeline, X, y, cv=loo, scoring='accuracy')

    # Collect predictions for ROC curve
    y_prob = np.zeros(len(y))
    for train_idx, test_idx in loo.split(X):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train = y[train_idx]

        # Train model and predict probability
        pipeline.fit(X_train, y_train)
        y_prob[test_idx] = pipeline.predict_proba(X_test)[:, 1]

    # Calculate ROC curve and AUC
    fpr, tpr, _ = roc_curve(y, y_prob)
    roc_auc = auc(fpr, tpr)

    # Plot ROC curve
    plt.figure()
    plt.plot(fpr, tpr, color='darkorange', lw=2,
             label=f'ROC curve (AUC = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title(f'ROC Curve for {label_col} Classification')
    plt.legend(loc="lower right")
    plt.savefig(f'./models/{label_col}_plot.png')
    plt.show()
    plt.close()

    # Report results
    mean_acc = np.mean(scores)
    std_acc = np.std(scores)
    print(f"[{label_col}] Leave-One-Out CV Accuracy: {mean_acc:.3f} (+/- {std_acc:.3f}), "
          f"over {len(scores)} folds (N={len(scores)})")
    print(f"[{label_col}] AUC-ROC Score: {roc_auc:.3f}")

    # Fit on ALL data to produce final model
    pipeline.fit(X, y)
    return pipeline



def main():
    # 1. Load the CSV - updated path to match project structure
    df = pd.read_csv("../data/behaviour_data.csv")
    
    print("Data preview:\n", df.head(), "\n")
    print("Total records:", len(df))

    # 2. Define which behavior labels you have
    behavior_cols = ['bark', 'growl', 'lick', "whine"]

    # Create models directory if it doesn't exist
    os.makedirs('./models', exist_ok=True)

    # 3. Train behavior models
    behavior_models = {}
    # Updated parameters for all behaviors
    model_params = {
        'bark': {
            'C': 0.2,
            'min_df': 0.005,
            'class_weight': 'balanced',
            'max_features': 3000
        },
        'whine': {
            'C': 0.1,
            'min_df': 0.005,
            'class_weight': 'balanced',
            'max_features': 3000
        },
        'lick': {
            'C': 0.2,
            'min_df': 0.005,
            'class_weight': 'balanced',
            'max_features': 3000
        },
        'growl': {
            'C': 0.15,
            'min_df': 0.005,
            'class_weight': 'balanced',
            'max_features': 3000
        }
    }

    for beh in behavior_cols:
        print(f"\n--- Training model for behavior: {beh} ---")
        params = model_params[beh]
        model = train_and_evaluate_loo(
            df,
            label_col=beh,
            penalty='l2',
            **params
        )
        behavior_models[beh] = model

        # Save each behavior model
        model_filename = f"./models/{beh}_model.pkl"
        dump(model, model_filename)
        print(f"Saved {beh} model to {model_filename}")


if __name__ == "__main__":
    main()