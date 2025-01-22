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
    'relevance_model.pkl',
    'bark_model.pkl',
    'growl_model.pkl',
    'lick_model.pkl',
    'whine_model.pkl',
    'scared_model.pkl',
    'angry_model.pkl',
    'anxious_model.pkl',
    'behavior_category_2_model.pkl',
    'behavior_category_3_model.pkl',
    'behavior_category_5_model.pkl'
]

behavior_models = ['bark_model.pkl', 'growl_model.pkl', 'lick_model.pkl', 'whine_model.pkl']
emotion_models = ['scared_model.pkl', 'angry_model.pkl', 'anxious_model.pkl', 'behavior_category_2_model.pkl', 'behavior_category_3_model.pkl', 'behavior_category_5_model.pkl']
thresholds = {
    'angry_model.pkl': 0.31,
    'anxious_model.pkl': 0.49,
    'bark_model.pkl': 0.495,
    'behavior_category_2_model.pkl': 0.46,
    'behavior_category_3_model.pkl': 0.48,
    'behavior_category_5_model.pkl': 0.29,
    'growl_model.pkl': 0.484,
    'lick_model.pkl': 0.47,
    'whine_model.pkl': 0.494,
    'relevance_model.pkl': 0.5,
    'scared_model.pkl': 0.44
}

data_file = "../data/reddit_animal_emotions1.json"

with open(data_file, 'r') as f:
    data = json.load(f)

def predict(post, model_path, emotion=""):
    try:
        # Load the model from joblib file
        model_full_path = f'../models/{model_path}'
        if not os.path.exists(model_full_path):
            raise FileNotFoundError(f"Model file not found: {model_full_path}")
            
        model = load(model_full_path)
        # Create text input by combining title and body
        if (emotion == ""):
            text = f"{post['body']}"
        else:
            text = f"{emotion} {post['body']} {emotion}"
        # Preprocess the text using the same function as during training

        # Return prediction probability - wrap processed_text in a list
        return model.predict_proba([text])[0,1] > thresholds[model_path]
    except Exception as e:
        print(f"Error loading or using model {model_path}: {str(e)}")
        raise

# csv file to store the results
csv_file = "results/reddit_animal_emotions_1.csv"
# Create results directory if it doesn't exist
os.makedirs(os.path.dirname(csv_file), exist_ok=True)

# Create a dictionary to store behavior-emotion counts
def create_empty_results():
    return {
        'bark': {emo.replace('_model.pkl', ''): 0 for emo in emotion_models},
        'growl': {emo.replace('_model.pkl', ''): 0 for emo in emotion_models},
        'lick': {emo.replace('_model.pkl', ''): 0 for emo in emotion_models},
        'whine': {emo.replace('_model.pkl', ''): 0 for emo in emotion_models}
    }

with open(csv_file, 'w', newline='') as f:
    writer = csv.writer(f)
    # Write header with emotions as columns
    header = ['behavior'] + [model.replace('_model.pkl', '') for model in emotion_models]
    writer.writerow(header)
    
    results = create_empty_results()
    
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
            text = f"{post['body']}"
            is_relevant = relevance_model.predict_proba([text])[0,1] > thresholds[relevance_model_path]
            
            # Only process relevant posts
            if is_relevant:
                # Check each behavior
                for behavior_model in behavior_models:
                    behavior_name = behavior_model.replace('_model.pkl', '')
                    if predict(post, behavior_model):
                        # If behavior is detected, check for emotions
                        for emotion_model in emotion_models:
                            emotion_name = emotion_model.replace('_model.pkl', '')
                            if predict(post, emotion_model, behavior_name):
                                results[behavior_name][emotion_name] += 1
        except Exception as e:
            print(f"Error processing post: {str(e)}")
            continue
                                
    # Write the results to CSV
    for behavior in results:
        row = [behavior] + [results[behavior][emo.replace('_model.pkl', '')] for emo in emotion_models]
        writer.writerow(row)
