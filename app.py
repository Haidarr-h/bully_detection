import streamlit as st
import mlflow
import os
import pandas as pd
from sklearn.linear_model import LogisticRegression
from mlflow.tracking import MlflowClient
import subprocess

# Set the MLflow tracking URI
os.environ['MLFLOW_TRACKING_URI'] = 'file:./mlruns'

# Function to get the latest run ID
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

# Load the latest model and vectorizer
run_id = get_latest_run_id_by_tag()
model_uri = f"runs:/{run_id}/logistic_regression_model"
vectorizer_uri = f"runs:/{run_id}/vectorizer"

print(f"Model URI: {model_uri}")
print(f"Vectorizer URI: {vectorizer_uri}")

model = mlflow.sklearn.load_model(model_uri)
vectorizer = mlflow.sklearn.load_model(vectorizer_uri)

# Function to retrieve model performance metrics from MLflow
def get_model_performance(run_id):
    try:
        client = MlflowClient()
        # Get metrics from the model run
        run = client.get_run(run_id)
        metrics = run.data.metrics
        return metrics
    except Exception as e:
        print(f"Error fetching model performance: {e}")
        return {}

# Function to count the number of rows in feedback data
def get_feedback_count():
    if os.path.exists('feedback_data.csv'):
        feedback_data = pd.read_csv('feedback_data.csv')
        return len(feedback_data)
    return 0

# Function to retrain the model
def retrain_model():
    subprocess.run(["docker", "build", "-t", "retrain_model", "."])  # Build Docker image
    subprocess.run(["docker", "run", "--rm", "retrain_model"])  # Run Docker container
    
    # After retraining, load the new model
    new_run_id = get_latest_run_id_by_tag()  # Get the new latest model run_id
    model_uri = f"runs:/{new_run_id}/logistic_regression_model"
    vectorizer_uri = f"runs:/{new_run_id}/vectorizer"
    new_model = mlflow.sklearn.load_model(model_uri)
    new_vectorizer = mlflow.sklearn.load_model(vectorizer_uri)

    return new_model, new_vectorizer

# Function to log feedback to a CSV file
def log_feedback_to_csv(text, prediction, feedback):
    fixed = prediction
    
    if feedback == 'No':
        if prediction == 'Toxic':
            fixed = 'Not Toxic'
        if prediction == 'Not Toxic':
            fixed = 'Toxic'
    
    new_row = {'text': text, 'prediction': prediction, 'feedback': feedback, 'fixed': fixed}
    
    # Append feedback to the CSV
    if not os.path.exists('feedback_data.csv'):
        pd.DataFrame([new_row]).to_csv('feedback_data.csv', index=False)
    else:
        feedback_data = pd.read_csv('feedback_data.csv')
        feedback_data = pd.concat([feedback_data, pd.DataFrame([new_row])])
        feedback_data.to_csv('feedback_data.csv', index=False)

    # Check if retraining is needed
    feedback_count = get_feedback_count()
    if feedback_count % 1 == 0:  # Retrain after every 5 new rows
        st.info("Retraining the model with new feedback data...")
        global model, vectorizer
        model, vectorizer = retrain_model()
        st.success("Model retrained successfully!")

# Function to make predictions
def predict(input_text):
    transformed_text = vectorizer.transform([input_text])
    prediction = model.predict(transformed_text)[0]
    return prediction

# Streamlit UI code
if "state" not in st.session_state:
    st.session_state.state = "default"

if "user_input" not in st.session_state:
    st.session_state.user_input = ""

if "prediction" not in st.session_state:
    st.session_state.prediction = None

if "feedback" not in st.session_state:
    st.session_state.feedback = None

# Tabs for Prediction and Model Performance
tab_selection = st.selectbox("Select a tab", ("Prediction", "Model Performance"))

if tab_selection == "Prediction":
    if st.session_state.state == "default":
        st.title('Cyberbullying Detection')
        st.subheader('Detect toxic language')
        st.session_state.user_input = st.text_area("Enter text:", "")

        if st.button("Predict"):
            if st.session_state.user_input:
                st.session_state.prediction = predict(st.session_state.user_input)
                st.session_state.state = "prediction"

    if st.session_state.state == "prediction":
        label = "Toxic" if st.session_state.prediction == 1 else "Not Toxic"
        st.success(f"Prediction: {label}")
        st.write("Does this prediction make sense?")
        col1, col2 = st.columns(2)
        with col1:
            if st.button("Yes"):
                st.session_state.feedback = "Yes"
                log_feedback_to_csv(st.session_state.user_input, label, "Yes")
                st.session_state.state = "feedback"
        with col2:
            if st.button("No"):
                st.session_state.feedback = "No"
                log_feedback_to_csv(st.session_state.user_input, label, "No")
                st.session_state.state = "feedback"

    if st.session_state.state == "feedback":
        if st.session_state.feedback == "Yes":
            st.success("Thank you for your feedback!")
        elif st.session_state.feedback == "No":
            st.write("Sorry for the confusion. We'll improve!")

        if st.button("Back to Default Page"):
            st.session_state.state = "default"
            st.session_state.user_input = ""
            st.session_state.prediction = None
            st.session_state.feedback = None

elif tab_selection == "Model Performance":
    st.title("Model Performance")
    
    # Get the performance of the current model
    metrics = get_model_performance(run_id)
    
    if metrics:
        st.write("### Model Metrics")
        st.write(f"**Accuracy**: {metrics.get('accuracy', 'N/A')}")
        st.write(f"**Precision**: {metrics.get('precision', 'N/A')}")
        st.write(f"**Recall**: {metrics.get('recall', 'N/A')}")
        st.write(f"**F1-Score**: {metrics.get('f1_score', 'N/A')}")
    else:
        st.write("No model performance data found.")
