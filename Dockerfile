# Use a lightweight base
FROM python:3.11-slim

WORKDIR /app

# Install dependencies only
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy your source code
COPY . .

# The app will download the model to this directory on the first start
# This directory should be part of a volume or just written to the ephemeral disk
ENV TRANSFORMERS_CACHE=/app/model_cache

CMD ["gunicorn", "app.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]