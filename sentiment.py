import pandas as pd
import numpy as np
import re
import joblib
from collections import Counter
import matplotlib.pyplot as plt
import seaborn as sns
import os
from functools import lru_cache

class MLSentimentAnalyzer:
    def __init__(self):
        self.model = None
        self.sentiment_mapping = None
        self.reverse_mapping = None
        self.load_model()
        
    def load_model(self):
        """Load the trained ML model and mappings"""
        try:
            if os.path.exists('sentiment_model.joblib'):
                self.model = joblib.load('sentiment_model.joblib')
                print("✅ ML Sentiment model loaded successfully!")
            else:
                print("❌ Model file not found. Please run train_sentiment_model.py first.")
                return False
                
            if os.path.exists('sentiment_mapping.joblib'):
                self.sentiment_mapping = joblib.load('sentiment_mapping.joblib')
                self.reverse_mapping = joblib.load('reverse_sentiment_mapping.joblib')
                print("✅ Sentiment mappings loaded successfully!")
            else:
                print("❌ Mapping files not found. Please run train_sentiment_model.py first.")
                return False
                
            return True
        except Exception as e:
            print(f"❌ Error loading model: {e}")
            return False

# Initialize the ML sentiment analyzer
ml_analyzer = MLSentimentAnalyzer()

# Dangerous/harmful keywords and patterns
DANGEROUS_KEYWORDS = [
    'kill', 'murder', 'suicide', 'harm', 'hurt', 'violence', 'threat', 'bomb', 
    'weapon', 'gun', 'knife', 'attack', 'fight', 'beat', 'destroy', 'hate',
    'bully', 'abuse', 'harassment', 'toxic', 'revenge', 'punish', 'eliminate'
]

def _load_hinglish_stopwords() -> set:
    """Load Hinglish/custom stopwords from stop_hinglish.txt if present."""
    stopwords = set()
    try:
        if os.path.exists('stop_hinglish.txt'):
            with open('stop_hinglish.txt', 'r', encoding='utf-8', errors='ignore') as f:
                # File may be newline or space separated; split on any whitespace
                for token in re.split(r"\s+", f.read().lower()):
                    token = token.strip()
                    if token:
                        stopwords.add(token)
    except Exception:
        # Fail-safe: just return what we have
        pass
    return stopwords

@lru_cache(maxsize=1)
def get_stopwords() -> set:
    """Build a robust stopword set (English + Hinglish + chat tokens)."""
    english_stopwords = {
        'i','me','my','myself','we','our','ours','ourselves','you','your','yours','yourself','yourselves',
        'he','him','his','himself','she','her','hers','herself','it','its','itself','they','them','their','theirs','themselves',
        'what','which','who','whom','this','that','these','those','am','is','are','was','were','be','been','being',
        'have','has','had','having','do','does','did','doing','a','an','the','and','but','if','or','because','as','until','while',
        'of','at','by','for','with','about','against','between','into','through','during','before','after','above','below',
        'to','from','up','down','in','out','on','off','over','under','again','further','then','once','here','there','when','where',
        'why','how','all','any','both','each','few','more','most','other','some','such','no','nor','not','only','own','same',
        'so','than','too','very','can','will','just','don','should','now','y','u','ok','okay','yeah','yup','nope','pls','please'
    }

    chat_fillers = {
        'guys','hey','hi','hello','hmm','hmmm','hahah','haha','lol','omg','btw','idk','imo','imho','btw','thx','thanks',
        'okie','k','kk','mm','umm','uh','ya','yaar','bro','bhai','sir','madam','dear'
    }

    # Merge sets
    stopwords = set()
    stopwords.update(english_stopwords)
    stopwords.update(chat_fillers)
    stopwords.update(_load_hinglish_stopwords())
    return stopwords

def clean_message(message):
    """Clean message for sentiment analysis (same as training preprocessing)"""
    if pd.isna(message) or message.strip() == '<Media omitted>':
        return ""
    
    # Convert to string and clean
    message = str(message).lower()
    # Remove URLs, mentions, and special characters
    message = re.sub(r'http\S+|www\S+|@\w+', '', message)
    message = re.sub(r'[^a-zA-Z\s]', ' ', message)
    message = re.sub(r'\s+', ' ', message).strip()
    
    return message

def analyze_sentiment_ml(message):
    """Analyze sentiment using the trained ML model"""
    if not ml_analyzer.model or not ml_analyzer.reverse_mapping:
        return {'sentiment': 'neutral', 'confidence': 0.0, 'polarity': 0}
    
    cleaned_msg = clean_message(message)
    if not cleaned_msg:
        return {'sentiment': 'neutral', 'confidence': 0.0, 'polarity': 0}
    
    try:
        # Make prediction
        prediction = ml_analyzer.model.predict([cleaned_msg])[0]
        
        # Get prediction probabilities for confidence (if available)
        try:
            probabilities = ml_analyzer.model.predict_proba([cleaned_msg])[0]
            confidence = max(probabilities)
        except AttributeError:
            # For models without predict_proba (like SVM), use decision function or default confidence
            try:
                decision_scores = ml_analyzer.model.decision_function([cleaned_msg])
                if isinstance(decision_scores, np.ndarray) and decision_scores.ndim > 1:
                    # Multi-class case - take the maximum absolute value
                    max_score = np.max(np.abs(decision_scores))
                else:
                    # Binary case
                    max_score = abs(decision_scores[0])
                # Normalize decision function to 0-1 range
                confidence = min(1.0, max(0.0, max_score / 2.0 + 0.5))
            except (AttributeError, IndexError):
                # Default confidence for models without probability estimates
                confidence = 0.7
        
        # Convert numeric prediction to sentiment label
        sentiment = ml_analyzer.reverse_mapping[prediction]
        
        # Calculate polarity based on sentiment
        if sentiment == 'positive':
            polarity = 0.5 + (confidence * 0.5)  # 0.5 to 1.0
        elif sentiment == 'negative':
            polarity = -0.5 - (confidence * 0.5)  # -1.0 to -0.5
        else:  # neutral
            polarity = 0.0
        
        return {
            'sentiment': sentiment,
            'confidence': confidence,
            'polarity': polarity
        }
    except Exception as e:
        print(f"Error in ML sentiment analysis: {e}")
        return {'sentiment': 'neutral', 'confidence': 0.0, 'polarity': 0}

def detect_dangerous_content(message):
    """Detect potentially dangerous or harmful content"""
    cleaned_msg = clean_message(message)
    if not cleaned_msg:
        return {'is_dangerous': False, 'danger_score': 0, 'matched_keywords': []}
    
    words = cleaned_msg.split()
    matched_keywords = []
    danger_score = 0
    
    for word in words:
        if word in DANGEROUS_KEYWORDS:
            matched_keywords.append(word)
            danger_score += 1
    
    # Additional pattern matching
    threat_patterns = [
        r'i will\s+\w*\s*(kill|hurt|harm)',
        r'going to\s+\w*\s*(kill|hurt|harm)',
        r'(kys|kill yourself)',
        r'(die|death)\s+(threat|wish)',
    ]
    
    for pattern in threat_patterns:
        if re.search(pattern, cleaned_msg):
            danger_score += 2
            matched_keywords.append('threat_pattern')
    
    is_dangerous = danger_score > 0
    
    return {
        'is_dangerous': is_dangerous,
        'danger_score': danger_score,
        'matched_keywords': matched_keywords
    }

def comprehensive_sentiment_analysis(selected_user, df):
    """Perform comprehensive sentiment analysis on chat data using ML model"""
    if selected_user != 'Overall':
        df = df[df['user'] == selected_user]

    # Filter out group notifications and media messages
    temp_df = df[df['user'] != 'group_notification'].copy()
    # Accept both '<Media omitted>' and '<Media omitted>\n'
    temp_df = temp_df[~temp_df['message'].astype(str).str.strip().eq('<Media omitted>')]
    temp_df = temp_df.dropna(subset=['message'])

    if temp_df.empty:
        return None
    
    # Analyze each message
    sentiments = []
    dangers = []
    
    print(f"Analyzing {len(temp_df)} messages using ML model...")
    
    for message in temp_df['message']:
        sentiment_result = analyze_sentiment_ml(message)
        danger_result = detect_dangerous_content(message)
        
        sentiments.append(sentiment_result)
        dangers.append(danger_result)
    
    # Create results dataframe
    results_df = temp_df.copy()
    results_df['polarity'] = [s['polarity'] for s in sentiments]
    results_df['confidence'] = [s['confidence'] for s in sentiments]
    results_df['sentiment'] = [s['sentiment'] for s in sentiments]
    results_df['is_dangerous'] = [d['is_dangerous'] for d in dangers]
    results_df['danger_score'] = [d['danger_score'] for d in dangers]
    results_df['matched_keywords'] = [d['matched_keywords'] for d in dangers]
    
    return results_df

def get_sentiment_summary(sentiment_df):
    """Get summary statistics of sentiment analysis"""
    if sentiment_df is None or sentiment_df.empty:
        return None
    
    total_messages = len(sentiment_df)
    dangerous_messages = sentiment_df['is_dangerous'].sum()
    
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    avg_polarity = sentiment_df['polarity'].mean()
    avg_confidence = sentiment_df['confidence'].mean()
    
    # Risk assessment
    danger_percentage = (dangerous_messages / total_messages) * 100
    if danger_percentage > 10:
        risk_level = "HIGH RISK"
    elif danger_percentage > 5:
        risk_level = "MEDIUM RISK"
    elif danger_percentage > 0:
        risk_level = "LOW RISK"
    else:
        risk_level = "SAFE"
    
    summary = {
        'total_messages': total_messages,
        'dangerous_messages': dangerous_messages,
        'danger_percentage': danger_percentage,
        'risk_level': risk_level,
        'sentiment_counts': sentiment_counts,
        'avg_polarity': avg_polarity,
        'avg_confidence': avg_confidence,
        'positive_messages': sentiment_counts.get('positive', 0),
        'negative_messages': sentiment_counts.get('negative', 0),
        'neutral_messages': sentiment_counts.get('neutral', 0)
    }
    
    return summary

def get_dangerous_messages(sentiment_df, limit=10):
    """Get the most dangerous messages for review"""
    if sentiment_df is None or sentiment_df.empty:
        return pd.DataFrame()
    
    dangerous_msgs = sentiment_df[sentiment_df['is_dangerous'] == True].copy()
    if dangerous_msgs.empty:
        return pd.DataFrame()
    
    # Sort by danger score and date
    dangerous_msgs = dangerous_msgs.sort_values(['danger_score', 'date'], ascending=[False, False])
    
    # Select relevant columns
    result = dangerous_msgs[['user', 'message', 'date', 'danger_score', 'matched_keywords']].head(limit)
    
    return result

def plot_sentiment_timeline(sentiment_df, selected_user):
    """Create sentiment timeline plot"""
    if sentiment_df is None or sentiment_df.empty:
        return None
    
    # Group by date and calculate average sentiment
    daily_sentiment = sentiment_df.groupby('only_date').agg({
        'polarity': 'mean',
        'sentiment': lambda x: x.value_counts().index[0]  # Most common sentiment
    }).reset_index()
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 8))
    
    # Plot 1: Polarity over time
    ax1.plot(daily_sentiment['only_date'], daily_sentiment['polarity'], 
             color='blue', marker='o', linewidth=2)
    ax1.axhline(y=0, color='white', linestyle='--', alpha=0.7)
    ax1.set_title(f'ML Sentiment Polarity Timeline - {selected_user}', fontsize=14)
    ax1.set_ylabel('Polarity (-1 to 1)')
    ax1.grid(True, alpha=0.3)
    
    # Plot 2: Sentiment distribution
    sentiment_counts = sentiment_df['sentiment'].value_counts()
    colors = {'positive': 'green', 'negative': 'red', 'neutral': 'gray'}
    ax2.bar(sentiment_counts.index, sentiment_counts.values, 
            color=[colors.get(x, 'blue') for x in sentiment_counts.index])
    ax2.set_title('ML Sentiment Distribution', fontsize=14)
    ax2.set_ylabel('Number of Messages')
    
    plt.tight_layout()
    return fig

def plot_danger_analysis(sentiment_df):
    """Create danger analysis visualization"""
    if sentiment_df is None or sentiment_df.empty:
        return None
    
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    # Plot 1: Safe vs Dangerous ratio
    safe_count = (~sentiment_df['is_dangerous']).sum()
    dangerous_count = sentiment_df['is_dangerous'].sum()
    
    labels = ['Safe Messages', 'Dangerous Messages']
    sizes = [safe_count, dangerous_count]
    colors = ['lightgreen', 'red']
    
    ax1.pie(sizes, labels=labels, colors=colors, autopct='%1.1f%%', startangle=90)
    if dangerous_count > 0:
        danger_scores = sentiment_df[sentiment_df['is_dangerous']]['danger_score']
        ax2.hist(danger_scores, bins=range(1, max(danger_scores)+2), 
                color='red', alpha=0.7, edgecolor='black')
        ax2.set_title('Danger Score Distribution', fontsize=14)
        ax2.set_xlabel('Danger Score')
        ax2.set_ylabel('Number of Messages')
    else:
        ax2.text(0.5, 0.5, 'No dangerous messages found!', 
                ha='center', va='center', transform=ax2.transAxes, fontsize=16)
        ax2.set_title('Danger Score Distribution', fontsize=14)
    
    plt.tight_layout()
    return fig

def get_keyword_analysis(sentiment_df):
    """Analyze most common keywords in different sentiment categories"""
    if sentiment_df is None or sentiment_df.empty:
        return {}
    
    analysis = {}
    
    for sentiment in ['positive', 'negative', 'neutral']:
        messages = sentiment_df[sentiment_df['sentiment'] == sentiment]['message']
        all_words = []
        
        for message in messages:
            cleaned = clean_message(message)
            for token in cleaned.split():
                # Filter tokens: remove stopwords, very short tokens, and numeric tokens
                if len(token) < 3:
                    continue
                if token.isnumeric():
                    continue
                if token in get_stopwords():
                    continue
                all_words.append(token)
        
        if all_words:
            word_freq = Counter(all_words).most_common(10)
            analysis[sentiment] = word_freq
        else:
            analysis[sentiment] = []
    
    return analysis

def get_confidence_analysis(sentiment_df):
    """Get confidence analysis for ML predictions"""
    if sentiment_df is None or sentiment_df.empty:
        return None
    
    avg_confidence = sentiment_df['confidence'].mean()
    high_confidence = (sentiment_df['confidence'] > 0.8).sum()
    medium_confidence = ((sentiment_df['confidence'] > 0.6) & (sentiment_df['confidence'] <= 0.8)).sum()
    low_confidence = (sentiment_df['confidence'] <= 0.6).sum()
    
    return {
        'avg_confidence': avg_confidence,
        'high_confidence': high_confidence,
        'medium_confidence': medium_confidence,
        'low_confidence': low_confidence
    }