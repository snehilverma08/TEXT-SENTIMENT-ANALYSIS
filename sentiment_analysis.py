# ---------------------------------------------------------------
# TEXT SENTIMENT ANALYSIS (Positive / Negative / Neutral)
# FULL PROGRAM – TRAINING + USER INPUT
# Tools: Python, NLTK, scikit-learn, Pandas
# ---------------------------------------------------------------

import pandas as pd
import nltk
import re

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.sentiment import SentimentIntensityAnalyzer

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report

# ---------------------------------------------------------------
# 1. DOWNLOAD REQUIRED NLTK DATA (ONE TIME)
# ---------------------------------------------------------------

nltk.download("stopwords")
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("wordnet")
nltk.download("vader_lexicon")

# ---------------------------------------------------------------
# 2. SAMPLE TRAINING DATA
# ---------------------------------------------------------------

data = {
    "text": [
        "I love this product",
        "This is the worst experience ever",
        "It is okay not good not bad",
        "Absolutely fantastic service",
        "I am very disappointed",
        "Nothing special it is fine",
        "Totally worth the money",
        "Terrible I want a refund",
        "I am very sad today",
        "I am extremely happy",
        "This ruined my day",
        "I am satisfied with the service",
        "Not happy with the support",
        "Amazing experience",
        "Very bad quality"
    ],
    "sentiment": [
        "positive","negative","neutral","positive","negative",
        "neutral","positive","negative","negative","positive",
        "negative","positive","negative","positive","negative"
    ]
}

df = pd.DataFrame(data)

# ---------------------------------------------------------------
# 3. TEXT PREPROCESSING
# ---------------------------------------------------------------

stop_words = set(stopwords.words("english"))
lemmatizer = WordNetLemmatizer()

def preprocess(text):
    text = text.lower()
    text = re.sub(r"[^a-zA-Z ]", "", text)
    tokens = nltk.word_tokenize(text)
    tokens = [lemmatizer.lemmatize(w) for w in tokens if w not in stop_words]
    return " ".join(tokens)

df["clean_text"] = df["text"].apply(preprocess)

# ---------------------------------------------------------------
# 4. TRAIN-TEST SPLIT
# ---------------------------------------------------------------

X_train, X_test, y_train, y_test = train_test_split(
    df["clean_text"], df["sentiment"], test_size=0.2, random_state=42
)

# ---------------------------------------------------------------
# 5. TF-IDF VECTORIZATION
# ---------------------------------------------------------------

vectorizer = TfidfVectorizer(ngram_range=(1,2))
X_train_tfidf = vectorizer.fit_transform(X_train)
X_test_tfidf = vectorizer.transform(X_test)

# ---------------------------------------------------------------
# 6. MODEL TRAINING
# ---------------------------------------------------------------

nb_model = MultinomialNB()
nb_model.fit(X_train_tfidf, y_train)

lr_model = LogisticRegression(max_iter=2000, class_weight="balanced")
lr_model.fit(X_train_tfidf, y_train)

# ---------------------------------------------------------------
# 7. MODEL EVALUATION
# ---------------------------------------------------------------

predictions = lr_model.predict(X_test_tfidf)

print("\nMODEL EVALUATION (LOGISTIC REGRESSION)")
print("Accuracy:", accuracy_score(y_test, predictions))
print(classification_report(y_test, predictions))

# ---------------------------------------------------------------
# 8. USER INPUT SENTIMENT ANALYSIS (ROBUST)
# ---------------------------------------------------------------

sia = SentimentIntensityAnalyzer()

def predict_sentiment(text):
    scores = sia.polarity_scores(text)
    compound = scores["compound"]

    if compound >= 0.05:
        return "POSITIVE"
    elif compound <= -0.05:
        return "NEGATIVE"
    else:
        return "NEUTRAL"

# ---------------------------------------------------------------
# 9. USER INTERACTION LOOP
# ---------------------------------------------------------------

print("\n----------------------------------------")
print(" TEXT SENTIMENT ANALYSIS SYSTEM ")
print(" Type 'exit' to quit ")
print("----------------------------------------")

while True:
    user_text = input("\nEnter a sentence: ")

    if user_text.lower() == "exit":
        print("Exiting program. Thank you!")
        break

    sentiment = predict_sentiment(user_text)
    print("Sentiment:", sentiment)
