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

def train_and_evaluate_loo(df, label_col, penalty='l2', C=1.0):
    X = df['post_content'].values
    y = df[label_col].values
    print(f"Training model for {label_col}...")

    # Define a pipeline: TF-IDF vectorization -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_clean_and_stem,
            ngram_range=(1, 3),  # Increased to capture more context
            min_df=0.005,        # Lowered to capture more rare but potentially important terms
            max_df=0.95,         # Added to remove very common terms
            max_features=5000    # Added to control vocabulary size
        )),
        ('clf', LogisticRegression(
            penalty=penalty,
            C=C,
            random_state=42,
            max_iter=1000,       # Added to ensure convergence
            class_weight='balanced'  # Added to handle class imbalance
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
    plt.savefig(f'../roc/{label_col}_roc_curve.png')
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

    # 1. Load the CSV
    # Adjust 'reddit_data.csv' to your actual file name/path
    df = pd.read_csv("../data/relevant_posts_with_categories.csv")

    print("Data preview:\n", df.head(), "\n")
    print("Total records:", len(df))

    # 2. Define which emotion labels you have
    # If you have more emotions, add them here
    emotion_cols = ['scared', 'anxious', 'angry', 'behavior_category_2', 'behavior_category_3', 'behavior_category_5']

    # Updated C values - increased regularization strength for 'anxious'
    c_values = {
        'scared': 1.0,
        'anxious': 0.1,  # Decreased C value to increase regularization
        'angry': 1.0,
        'behavior_category_2': 1.0,
        'behavior_category_3': 1.0,
        'behavior_category_5': 1.0
    }

    # 3. Train emotion models (one per emotion) with LOO CV
    emotion_models = {}
    for emo in emotion_cols:
        print(f"\n--- Training model for emotion: {emo} ---")
        if emo == 'anxious':
            # Special pipeline configuration for anxious classification
            model = Pipeline([
                ('tfidf', TfidfVectorizer(
                    tokenizer=tokenize_clean_and_stem,
                    ngram_range=(1, 2),  # Reduced to prevent overfitting
                    min_df=0.02,         # Increased to focus on more common terms
                    max_df=0.8,          # Stricter upper bound
                    max_features=3000    # Reduced features
                )),
                ('clf', LogisticRegression(
                    penalty='l2',
                    C=c_values[emo],
                    random_state=42,
                    max_iter=1000,
                    class_weight='balanced'
                ))
            ])
            model = train_and_evaluate_loo(df, label_col=emo, penalty='l2', C=c_values[emo])
        else:
            model = train_and_evaluate_loo(df, label_col=emo, penalty='l2', C=c_values[emo])
        emotion_models[emo] = model

        # Save each emotion model
        model_filename = f"../models/{emo}_model.pkl"
        dump(model, model_filename)
        print(f"Saved {emo} model to {model_filename}")


if __name__ == "__main__":
    main()