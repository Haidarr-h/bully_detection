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
feedback_data = pd.read_csv('feedback_data.csv')

# Mengubah nilai 'fixed' menjadi 0 dan 1
feedback_data['fixed'] = feedback_data['fixed'].map({'Toxic': 1, 'Not Toxic': 0})

# Menggabungkan dataset sesuai dengan fitur dan target
X = pd.concat([df['Text'], feedback_data['text']], axis=0).reset_index(drop=True)
y = pd.concat([df['oh_label'], feedback_data['fixed']], axis=0).reset_index(drop=True)

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
    model = LogisticRegression(max_iter=1000)
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
    mlflow.sklearn.log_model(model, "logistic_regression_model", input_example=X_train_tfidf[0].toarray())


    # Log the vectorizer with example input
    mlflow.sklearn.log_model(
        tfidf,
        "vectorizer"
    )

    # Print the classification report for inspection
    print(f'Accuracy: {accuracy:.4f}')
    print('Classification Report:')
    print(classification_report(y_test, y_pred))
