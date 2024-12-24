import mlflow
import mlflow.sklearn
import pandas as pd
import os
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
import mlflow.pyfunc


# Set Tracking URI
mlflow.set_tracking_uri("file://" + os.path.abspath("mlruns"))

print('=================================')
print('Retraining Model')

def update_is_latest_tag(experiment_id, run_id, is_latest_value):
    client = MlflowClient()
    runs = client.search_runs(
        experiment_ids=experiment_id,
        filter_string="tags.is_latest = 'True'",
        max_results=1
    )
    for r in runs:
        client.set_tag(r.info.run_id, "is_latest", "False")
    client.set_tag(run_id, "is_latest", is_latest_value)

def get_latest_run_id_by_tag():
    try:
        client = MlflowClient()
        # Ambil semua run dari eksperimen tertentu (sesuaikan ID eksperimen Anda)
        runs = client.search_runs(
            experiment_ids=["228268414314891703"],  # Ganti dengan ID eksperimen Anda
            order_by=["attributes.start_time DESC"],  # Urutkan berdasarkan waktu mulai
            max_results=1
        )
        if runs:
            return runs[0].info.run_id
    except Exception as e:
        print(f"Error fetching latest run: {e}")
    return '1f1144e3e5ab4710ae90c52e2a1a936d'

experiment_id = "228268414314891703"
previous_run_id = get_latest_run_id_by_tag()

if previous_run_id is None:
    raise ValueError("No latest run found. Please ensure a valid experiment and run exists.")

# Konversi URI model dan vectorizer ke path lokal
model_uri = f"mlruns/{experiment_id}/{previous_run_id}/artifacts/logistic_regression_model"
vectorizer_uri = f"mlruns/{experiment_id}/{previous_run_id}/artifacts/vectorizer"

# model_uri = f"runs:/{previous_run_id}/logistic_regression_model"
# vectorizer_uri = f"runs:/{previous_run_id}/vectorizer"

model_uri = os.path.abspath(model_uri)
vectorizer_uri = os.path.abspath(vectorizer_uri)

print(f"Model URI: {model_uri}")
print(f"Vectorizer URI: {vectorizer_uri}")

# Debug untuk melihat isi direktori
model_dir = os.path.dirname(model_uri)
vectorizer_dir = os.path.dirname(vectorizer_uri)

print(f"Model Directory: {model_dir}")
print(f"Vectorizer Directory: {vectorizer_dir}")

if os.path.exists(model_dir):
    print(f"Contents of {model_dir}: {os.listdir(model_dir)}")
else:
    print(f"Directory {model_dir} does not exist.")

if os.path.exists(vectorizer_dir):
    print(f"Contents of {vectorizer_dir}: {os.listdir(vectorizer_dir)}")
else:
    print(f"Directory {vectorizer_dir} does not exist.")

try:
    model = mlflow.sklearn.load_model(model_uri)
    print(model)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")

try:
    vectorizer = mlflow.sklearn.load_model(vectorizer_uri)
    print(vectorizer)
    print("Vectorizer loaded successfully!")
except Exception as e:
    print(f"Error loading vectorizer: {e}")

# # Load model dan vectorizer
# model = mlflow.sklearn.load_model(model_uri)
# # model = mlflow.pyfunc.load_model(model_uri)
# vectorizer = mlflow.sklearn.load_model(vectorizer_uri)

# Retrain dengan data feedback
feedback_data = pd.read_csv('feedback_data.csv')
X_feedback = vectorizer.transform(feedback_data['text'])
y_feedback = feedback_data['fixed']

new_model = LogisticRegression()
new_model.fit(X_feedback, y_feedback)

print('sampai sini')

with mlflow.start_run():
    print('sampai sini1')
    
    mlflow.sklearn.log_model(new_model, "logistic_regression_model")
    print('sampai sini2')
    
    mlflow.sklearn.log_model(vectorizer, "vectorizer")
    new_run_id = mlflow.active_run().info.run_id
    update_is_latest_tag(experiment_id, new_run_id, "True")
    print("Model retrained and tagged as the latest model successfully.")
