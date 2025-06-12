import pandas as pd
import re
import string
import joblib
import argparse
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix

# -----------------------------
# Dataset Info:
# Due to size, the dataset is not included here.
# Download it from: https://www.kaggle.com/datasets/akshitamishra9204/real-and-fake-dataset
# and place 'combined_news.csv' in this folder.
# -----------------------------

# -----------------------------
# Step 1: Clean the text
# -----------------------------
def clean_text(text):
    if pd.isnull(text):
        return ""
    text = text.lower()
    text = re.sub(r'\[.*?\]', '', text)
    text = re.sub(r'https?://\S+|www\.\S+', '', text)
    text = re.sub(r'<.*?>+', '', text)
    text = re.sub(f"[{re.escape(string.punctuation)}]", '', text)
    text = re.sub(r'\n', ' ', text)
    text = re.sub(r'\w*\d\w*', '', text)
    return text.strip()

# -----------------------------
# Step 2: Load and preprocess data
# -----------------------------
def load_and_prepare_data(filepath):
    df = pd.read_csv(filepath)
    df['text'] = df['text'].astype(str)
    df['clean_text'] = df['text'].apply(clean_text)
    return df

# -----------------------------
# Step 3: Train Model
# -----------------------------
def train_model(df):
    X = df['clean_text']
    y = df['label']

    vectorizer = TfidfVectorizer(stop_words='english', max_df=0.7)
    X_vectorized = vectorizer.fit_transform(X)

    X_train, X_test, y_train, y_test = train_test_split(
        X_vectorized, y, test_size=0.2, random_state=42, stratify=y
    )

    model = LogisticRegression(max_iter=1000)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    print("Accuracy:", accuracy_score(y_test, y_pred))
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Confusion Matrix:\n", confusion_matrix(y_test, y_pred))

    return model, vectorizer

# -----------------------------
# Step 4: Save model and vectorizer
# -----------------------------
def save_model(model, vectorizer):
    joblib.dump(model, 'fake_news_model.pkl')
    joblib.dump(vectorizer, 'tfidf_vectorizer.pkl')
    print("Model and vectorizer saved!")

# -----------------------------
# Step 5: Predict single text
# -----------------------------
def predict_single(text, model_path='fake_news_model.pkl', vec_path='tfidf_vectorizer.pkl'):
    model = joblib.load(model_path)
    vectorizer = joblib.load(vec_path)
    cleaned = clean_text(text)
    vectorized = vectorizer.transform([cleaned])
    prediction = model.predict(vectorized)[0]
    return 'REAL' if prediction == 1 else 'FAKE'

# -----------------------------
# Main Execution
# -----------------------------
if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fake News Detection")
    parser.add_argument('--data', type=str, help='Path to combined_news.csv', default='combined_news.csv')
    parser.add_argument('--predict', type=str, help='Text to classify as real or fake', default=None)
    args = parser.parse_args()

    if args.predict:
        label = predict_single(args.predict)
        print("Prediction:", label)
    else:
        print("Loading and preparing data...")
        df = load_and_prepare_data(args.data)
        print("Training model...")
        model, vectorizer = train_model(df)
        save_model(model, vectorizer)

        # Only run feature importance after training
        feature_names = vectorizer.get_feature_names_out()
        coefs = model.coef_[0]
        top_features = np.argsort(coefs)

        print("Most fake-like words:", feature_names[top_features[-10:]])
        print("Most real-like words:", feature_names[top_features[:10]])
