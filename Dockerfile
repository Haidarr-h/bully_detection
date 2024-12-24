# Gunakan image dasar Python yang ringan
FROM python:3.9-slim

# Set working directory di dalam container
WORKDIR /app

# Copy requirements.txt ke dalam container
COPY requirements.txt .

# Install dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Salin semua file ke dalam container
COPY . .

# Set environment variable untuk MLflow tracking URI
# Anda dapat menggunakan direktori lokal untuk menyimpan artefak MLflow
ENV MLFLOW_TRACKING_URI=file:/app/mlruns

# Pastikan direktori 'mlruns' ada untuk menyimpan artefak
RUN mkdir -p /app/mlruns

# Command untuk menjalankan retraining
CMD ["python", "retrain_model.py"]
