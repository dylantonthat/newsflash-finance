#  NLP logic & training
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

def vectorize_text(titles):
    vectorizer = TfidfVectorizer(stop_words='english')
    return vectorizer.fit_transform(titles), vectorizer

def label_stock_movement(stock_df):
    stock_df['Movement'] = stock_df['Close'] > stock_df['Open']
    return stock_df

def train_model(X, y):
    model = LogisticRegression().fit(X, y)
    return model


def predict_sentiment(headlines, model, vectorizer):
    X_new = vectorizer.transform(headlines)
    
    # Headline Impact Confidence Score (extra credit)
    probs = model.predict_proba(X_new)
    preds = model.predict(X_new)
    return preds, probs
