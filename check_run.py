from mlflow.tracking import MlflowClient

client = MlflowClient()
experiment_id = "1"  # Sesuaikan dengan eksperimen Anda
runs = client.search_runs(experiment_ids=[experiment_id])

print('Test')

for run in runs:
    print(f"Run ID: {run.info.run_id}, Status: {run.info.status}, Start Time: {run.info.start_time}")