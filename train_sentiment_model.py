#!/usr/bin/env python3
"""
Machine Learning Sentiment Analysis Model Training Script
Trains a sentiment analysis model using the provided dataset and exports it using joblib
"""

import pandas as pd
import numpy as np
import re
import joblib
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
from sklearn.pipeline import Pipeline
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

class SentimentModelTrainer:
    def __init__(self):
        self.model = None
        self.vectorizer = None
        self.label_encoder = None
        
    def clean_text(self, text):
        """Clean and preprocess text data"""
        if pd.isna(text):
            return ""
        
        # Convert to string and lowercase
        text = str(text).lower()
        
        # Remove URLs
        text = re.sub(r'http\S+|www\S+|@\w+', '', text)
        
        # Remove special characters and digits
        text = re.sub(r'[^a-zA-Z\s]', '', text)
        
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text).strip()
        
        return text
    
    def load_and_preprocess_data(self):
        """Load and preprocess the training data"""
        print("Loading training data...")
        
        # Load training data with proper encoding handling
        try:
            train_df = pd.read_csv('dataset/train.csv', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                train_df = pd.read_csv('dataset/train.csv', encoding='latin-1')
            except UnicodeDecodeError:
                train_df = pd.read_csv('dataset/train.csv', encoding='cp1252')
        
        try:
            test_df = pd.read_csv('dataset/test.csv', encoding='utf-8')
        except UnicodeDecodeError:
            try:
                test_df = pd.read_csv('dataset/test.csv', encoding='latin-1')
            except UnicodeDecodeError:
                test_df = pd.read_csv('dataset/test.csv', encoding='cp1252')
        
        print(f"Training data shape: {train_df.shape}")
        print(f"Test data shape: {test_df.shape}")
        
        # Combine train and test data for better model performance
        combined_df = pd.concat([train_df, test_df], ignore_index=True)
        
        # Clean the text data
        print("Cleaning text data...")
        combined_df['cleaned_text'] = combined_df['text'].apply(self.clean_text)
        
        # Remove empty texts
        combined_df = combined_df[combined_df['cleaned_text'].str.len() > 0]
        
        # Check sentiment distribution
        print("\nSentiment distribution:")
        print(combined_df['sentiment'].value_counts())
        
        # Map sentiment labels to numeric values
        sentiment_mapping = {'positive': 1, 'negative': -1, 'neutral': 0}
        combined_df['sentiment_numeric'] = combined_df['sentiment'].map(sentiment_mapping)
        
        # Remove any rows with NaN sentiment
        combined_df = combined_df.dropna(subset=['sentiment_numeric'])
        
        print(f"\nFinal dataset shape: {combined_df.shape}")
        print(f"Sentiment distribution after cleaning:")
        print(combined_df['sentiment_numeric'].value_counts())
        
        return combined_df
    
    def train_models(self, X_train, y_train, X_test, y_test):
        """Train multiple models and select the best one"""
        models = {
            'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
            'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
            'SVM': SVC(kernel='linear', random_state=42)
        }
        
        best_model = None
        best_score = 0
        best_name = ""
        
        print("\nTraining and evaluating models...")
        
        for name, model in models.items():
            print(f"\nTraining {name}...")
            
            # Create pipeline with TF-IDF vectorizer
            pipeline = Pipeline([
                ('tfidf', TfidfVectorizer(max_features=10000, ngram_range=(1, 2))),
                ('classifier', model)
            ])
            
            # Train the model
            pipeline.fit(X_train, y_train)
            
            # Make predictions
            y_pred = pipeline.predict(X_test)
            
            # Calculate accuracy
            accuracy = accuracy_score(y_test, y_pred)
            print(f"{name} Accuracy: {accuracy:.4f}")
            
            # Store best model
            if accuracy > best_score:
                best_score = accuracy
                best_model = pipeline
                best_name = name
        
        print(f"\nBest model: {best_name} with accuracy: {best_score:.4f}")
        
        # Print detailed classification report for best model
        y_pred_best = best_model.predict(X_test)
        print(f"\nDetailed Classification Report for {best_name}:")
        print(classification_report(y_test, y_pred_best))
        
        return best_model, best_name
    
    def train(self):
        """Main training function"""
        print("=== Sentiment Analysis Model Training ===")
        
        # Load and preprocess data
        df = self.load_and_preprocess_data()
        
        # Prepare features and target
        X = df['cleaned_text']
        y = df['sentiment_numeric']
        
        # Split the data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42, stratify=y
        )
        
        print(f"\nTraining set size: {len(X_train)}")
        print(f"Test set size: {len(X_test)}")
        
        # Train models and select best one
        self.model, model_name = self.train_models(X_train, y_train, X_test, y_test)
        
        # Save the model
        model_filename = 'sentiment_model.joblib'
        joblib.dump(self.model, model_filename)
        print(f"\nModel saved as: {model_filename}")
        
        # Create a simple test
        print("\n=== Model Testing ===")
        test_texts = [
            "I love this app! It's amazing!",
            "This is terrible, I hate it.",
            "The weather is okay today.",
            "Great job on the project!",
            "I'm so disappointed with this service."
        ]
        
        predictions = self.model.predict(test_texts)
        sentiment_labels = {-1: 'negative', 0: 'neutral', 1: 'positive'}
        
        print("\nTest predictions:")
        for text, pred in zip(test_texts, predictions):
            print(f"Text: '{text}' -> Sentiment: {sentiment_labels[pred]}")
        
        return self.model
    
    def create_sentiment_mapping(self):
        """Create and save sentiment mapping"""
        sentiment_mapping = {
            'positive': 1,
            'negative': -1,
            'neutral': 0
        }
        
        # Reverse mapping for predictions
        reverse_mapping = {v: k for k, v in sentiment_mapping.items()}
        
        # Save mappings
        joblib.dump(sentiment_mapping, 'sentiment_mapping.joblib')
        joblib.dump(reverse_mapping, 'reverse_sentiment_mapping.joblib')
        
        print("Sentiment mappings saved successfully!")
        return sentiment_mapping, reverse_mapping

def main():
    """Main function to run the training"""
    trainer = SentimentModelTrainer()
    
    # Train the model
    model = trainer.train()
    
    # Create and save sentiment mappings
    trainer.create_sentiment_mapping()
    
    print("\n=== Training Complete ===")
    print("Files created:")
    print("- sentiment_model.joblib (trained model)")
    print("- sentiment_mapping.joblib (label to numeric mapping)")
    print("- reverse_sentiment_mapping.joblib (numeric to label mapping)")

if __name__ == "__main__":
    main()
