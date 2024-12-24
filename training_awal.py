import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, precision_score, recall_score, f1_score
import mlflow
import mlflow.sklearn
import numpy as np
import scipy
import sklearn

# Load dataset
df = pd.read_csv('dataset/attack_parsed_dataset.csv')

# Select relevant columns
X = df['Text']  # Feature: text
y = df['oh_label']  # Target: oh_label (1 for attack, 0 for not attack)

# Split the data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Convert text data into TF-IDF features
tfidf = TfidfVectorizer(stop_words='english', max_features=5000)
X_train_tfidf = tfidf.fit_transform(X_train)
X_test_tfidf = tfidf.transform(X_test)

# Set MLflow tracking URI and experiment
# mlflow.set_tracking_uri("http://127.0.0.1:5000")  # Alamat server MLflow lokal
mlflow.set_experiment("Cyberbullying Detection Experiment")

with mlflow.start_run():
    # Add compatibility tags for reproducibility
    mlflow.log_param("numpy_version", np.__version__)
    mlflow.log_param("pandas_version", pd.__version__)
    mlflow.log_param("scikit_learn_version", sklearn.__version__)
    mlflow.log_param("scipy_version", scipy.__version__)

    # Define and train a simple Logistic Regression model
    model = LogisticRegression()
    model.fit(X_train_tfidf, y_train)

    # Make predictions
    y_pred = model.predict(X_test_tfidf)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)

    # Log parameters
    mlflow.log_param("model", "LogisticRegression")
    mlflow.log_param("max_features", 5000)
    mlflow.log_param("test_size", 0.2)

    # Log metrics
    mlflow.log_metric("accuracy", accuracy)
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1)

    # Log the model with input example
    mlflow.sklearn.log_model(
        model,
        "logistic_regression_model",
        input_example=X_test_tfidf[0].toarray()
    )

    # Log the vectorizer with example input
    mlflow.sklearn.log_model(
        tfidf,
        "vectorizer",
        input_example=[X_test.iloc[0]]
    )

    # Print the classification report for inspection
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
