import os
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import classification_report, accuracy_score
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
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

# SBERT model to get sentence embeddings
sbert_model = SentenceTransformer('paraphrase-MiniLM-L6-v2')

# Function to get embeddings
def get_embeddings(texts):
    return sbert_model.encode(texts)

# Transform the text to embeddings
X_train_embeddings = get_embeddings(X_train.reset_index(drop=True))
X_test_embeddings = get_embeddings(X_test.reset_index(drop=True))

# Pipeline with StandardScaler + LogisticRegression
pipeline = Pipeline([
    ("scaler", StandardScaler()),  # Standardize features
    ("clf", LogisticRegression(max_iter=1000))
])

# Grid search parameter space
param_grid = {
    "clf__C": [0.1, 1.0, 10],
    "clf__solver": ["lbfgs", "liblinear"],
}

# Run grid search
grid = GridSearchCV(pipeline, param_grid, cv=5, n_jobs=-1, verbose=1)
grid.fit(X_train_embeddings, y_train)

# Evaluation
y_pred = grid.predict(X_test_embeddings)
print(f"\nBest Parameters: {grid.best_params_}")
print(f"Accuracy: {accuracy_score(y_test, y_pred):.4f}")
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Save the model
joblib.dump(grid.best_estimator_, "LR-SBERT.joblib")
print("\nModel saved as LR-SBERT.joblib")
