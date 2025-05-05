from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import classification_report

# Vectorize headlines using TF-IDF with unigrams and bigrams
def vectorize_text(titles):
    vectorizer = TfidfVectorizer(
        stop_words='english',
        ngram_range=(1, 2),         # Capture unigrams and bigrams
        max_features=1000           # Limit number of features for efficiency
    )
    return vectorizer.fit_transform(titles), vectorizer

# Label stock movement as binary outcome: 1 if stock rose, else 0
def label_stock_movement(stock_df):
    stock_df['Movement'] = stock_df['Close'] > stock_df['Open']
    return stock_df

# Train model using RandomForestClassifier with hyperparameter tuning
def train_model(X, y):
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20]
    }
    rf = RandomForestClassifier(random_state=42)
    grid = GridSearchCV(rf, param_grid, cv=5, scoring='accuracy')
    grid.fit(X, y)

    # Optional: Print model evaluation using train/test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    y_pred = grid.best_estimator_.predict(X_test)
    print("Model Evaluation on Held-Out Set:")
    print(classification_report(y_test, y_pred))

    return grid.best_estimator_

# Predict sentiment with confidence probabilities
def predict_sentiment(headlines, model, vectorizer):
    X_new = vectorizer.transform(headlines)
    probs = model.predict_proba(X_new)  # Get probability scores
    preds = model.predict(X_new)        # Get class predictions
    return preds, probs
