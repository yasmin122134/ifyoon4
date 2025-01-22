import json
import csv
import pickle
import os
import re
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
from nltk.corpus import stopwords
from joblib import load

# Download required NLTK data
nltk.download('punkt')
nltk.download('stopwords')

# Add the text processing function that was used during training
def tokenize_clean_and_stem(text):
    # Convert to lowercase
    text = text.lower()
    # Remove special characters and digits
    text = re.sub(r'[^a-zA-Z\s]', '', text)
    # Tokenize
    tokens = word_tokenize(text)
    # Remove stopwords
    stop_words = set(stopwords.words('english'))
    tokens = [token for token in tokens if token not in stop_words]
    # Stem
    stemmer = PorterStemmer()
    tokens = [stemmer.stem(token) for token in tokens]
    return ' '.join(tokens)

models = [
    'angry_model.pkl',
    'anxious_model.pkl',
    'bark_model.pkl',
    'behavior_category_2_model.pkl',
    'behavior_category_3_model.pkl',
    'behavior_category_5_model.pkl',
    'chew_model.pkl',
    'excited_model.pkl',
    'pant_model.pkl',
    'relevance_model.pkl',
    'scared_model.pkl'
]

thresholds = {
    'angry_model.pkl': 0.31,
    'anxious_model.pkl': 0.49,
    'bark_model.pkl': 0.42,
    'behavior_category_2_model.pkl': 0.46,
    'behavior_category_3_model.pkl': 0.48,
    'behavior_category_5_model.pkl': 0.29,
    'chew_model.pkl': 0.5,
    'excited_model.pkl': 0.5,
    'pant_model.pkl': 0.5,
    'scared_model.pkl': 0.5
}

data_file = "../data/reddit_animal_emotions1.json"

with open(data_file, 'r') as f:
    data = json.load(f)

def predict(post, model_path):
    try:
        # Load the model from joblib file
        model_full_path = f'../models/{model_path}'
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file not found: {model_full_path}")
            
        model = load(model_full_path)
        # Create text input using post_content instead of body
        text = f"{post['post_content']}"
        # Preprocess the text using the same function as during training
        processed_text = tokenize_clean_and_stem(text)
        # Return prediction probability - wrap processed_text in a list
        return model.predict_proba([text])[0,1]
    except Exception as e:
        print(f"Error loading or using model {model_path}: {str(e)}")
        raise

# csv file to store the results
csv_file = "results/reddit_animal_emotions_1.csv"
# Create results directory if it doesn't exist
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header using model names without '_model.pkl'
    header = ['post_id'] + [model.replace('_model.pkl', '') for model in models]
    writer.writerow(header)
    
    # Cache relevance model to avoid loading it multiple times
    relevance_model_path = 'relevance_model.pkl'
    try:
        model_full_path = f'../models/{relevance_model_path}'
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Relevance model file not found: {model_full_path}")
        
        relevance_model = load(model_full_path)
    except Exception as e:
        print(f"Error loading relevance model: {str(e)}")
        print(f"Please ensure the model file exists at: {model_full_path}")
        exit(1)
        
    for post in data:
        try:
            # Check relevance first
            text = f"{post['post_content']}"
            processed_text = tokenize_clean_and_stem(text)
            is_relevant = relevance_model.predict_proba([text])[0,1]
            
            # Only process relevant posts
            if True:
                # Create a row with post_id first
                row = [post['title']]  # Using title as ID for now
                # Get predictions for each model in order
                for model in models:
                    if model != relevance_model_path:
                        prediction = predict(post, model)
                        row.append(prediction)
                    else:
                        row.append(is_relevant)
                writer.writerow(row)
            
        except Exception as e:
            print(f"Error processing post {post['title']}: {str(e)}")
            continue
