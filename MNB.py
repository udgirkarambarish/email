import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import classification_report, accuracy_score
import joblib

# Load and label dataset
def load_and_label_data(filepath, label):
    df = pd.read_csv(filepath)
    df['label'] = label
    df['text'] = df['Subject'].fillna('') + ' ' + df['Body'].fillna('')
    return df[['text', 'label']]

# Combine all labeled data
dataframes = [
    load_and_label_data("dataset/preprocessed_spam.csv", "spam"),
    load_and_label_data("dataset/preprocessed_promotions.csv", "promotions"),
    load_and_label_data("dataset/preprocessed_updates.csv", "updates"),
    load_and_label_data("dataset/preprocessed_social.csv", "social"),
]
full_df = pd.concat(dataframes, ignore_index=True)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    full_df["text"], full_df["label"], test_size=0.2, random_state=42, stratify=full_df["label"]
)

# Pipeline setup
pipeline = Pipeline([
    ("tfidf", TfidfVectorizer(max_features=5000)),
    ("clf", MultinomialNB())
])

# Grid search parameter space
param_grid = {
    "tfidf__ngram_range": [(1, 1), (1, 2)],
    "tfidf__stop_words": [None, "english"],
    "tfidf__use_idf": [True, False],
    "clf__alpha": [0.5, 1.0, 1.5]
}

# Run grid search
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train, y_train)

# Evaluation
y_pred = grid.predict(X_test)
print(f"\nBest Parameters: {grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save model
joblib.dump(grid.best_estimator_, "email_classifier_model.joblib")
print("\nBest model saved as email_classifier_model.joblib")
