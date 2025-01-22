import warnings

# Suppress specific warning
warnings.filterwarnings("ignore",
                        message="The parameter 'token_pattern' will not be used since 'tokenizer' is not None")

import nltk
import sklearn
import pandas as pd
import numpy as np

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import LeaveOneOut, cross_val_score
from joblib import dump, load
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

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


def train_and_evaluate_loo(df, label_col, penalty='l2', C=1.0):
    X = df['post_content'].values
    y = df[label_col].values

    # Define a pipeline: TF-IDF vectorization -> Logistic Regression
    pipeline = Pipeline([
        ('tfidf', TfidfVectorizer(
            tokenizer=tokenize_clean_and_stem,
            ngram_range=(1, 4),  # Increased to include 4-grams
            min_df=1,  # Reduced to include rare but potentially important terms
            max_df=0.9,  # Slightly more aggressive removal of common terms
            max_features=15000,  # Increased vocabulary size
            sublinear_tf=True,
            use_idf=True,
            smooth_idf=True,
            norm='l2'
        )),
        ('clf', LogisticRegression(
            penalty=penalty,
            C=0.5,  # Increased regularization
            random_state=42,
            class_weight='balanced',
            max_iter=2000,
            solver='saga',  # Better for large datasets
            n_jobs=-1  # Parallel processing
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
    plt.savefig(f'{label_col}_roc_curve.png')
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
    df = pd.read_csv("../data/behaviour_data.csv")

    # Check columns exist (customize to your data)
    # Example columns: post_text, anger_label, joy_label, relevance_label
    print("Data preview:\n", df.head(), "\n")
    print("Total records:", len(df))

    # 4. Train relevance model
    # If your data has a 'relevance_label' column
    if 'relevance' in df.columns:
        print("\n=== Training Relevance Model ===")
        relevance_model = train_and_evaluate_loo(df, label_col='relevance', penalty='l2')

        # Save relevance model
        dump(relevance_model, "../models/relevance_model.pkl")
        print("Saved relevance model to relevance_model.pkl")
    else:
        relevance_model = None
        print("\nNo 'relevance_label' column found. Skipping relevance model training.\n")

    print("\nDone. All models have been trained with Leave-One-Out CV, "
          "final pipelines saved to disk, and example inference shown.")


if __name__ == "__main__":
    main()
