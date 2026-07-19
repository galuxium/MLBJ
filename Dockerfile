FROM python:3.11-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
# This downloads the model during the BUILD phase
RUN python -c "from sentence_transformers import CrossEncoder; CrossEncoder('law-ai/InCaseLawBERT')"
CMD ["gunicorn", "app.main:app", "-w", "1", "-k", "uvicorn.workers.UvicornWorker", "--bind", "0.0.0.0:8000"]