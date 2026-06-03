# Judicio AI - Backend API

Judicio AI is an advanced legal document retrieval, case prediction, summarization, and chat backend built with **FastAPI**. It leverages a state-of-the-art **Hybrid Search** engine combining Dense Vector Embeddings (Qdrant / BGE-large) and Sparse Keyword Search (BM25), dynamically re-ranked via a Cross-Encoder for highly accurate legal document retrieval.

## 🏗 Tech Stack

- **Framework:** FastAPI (Python)
- **Primary Database:** MongoDB (Truth store & metadata)
- **Vector Engine:** Qdrant
- **Machine Learning / AI:** 
  - Embeddings: `BAAI/bge-large-en-v1.5`
  - Reranker: `BAAI/bge-reranker-v2-m3`
- **Package Management:** `pip`

---

## 📋 Prerequisites

Before you start, ensure you have the following installed:
- **Python 3.9+** (3.10/3.11 recommended)
- **MongoDB** instance (Local or Atlas)
- **Qdrant** instance (Local via Docker `docker run -p 6333:6333 qdrant/qdrant` or Cloud)

---

## 🚀 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd judicio-ai-backend
```

### 2. Set Up a Virtual Environment
We recommend using a virtual environment to isolate project dependencies.
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

---

## ⚙️ Environment Variables

Create a `.env` file in the root directory and populate it with the necessary configuration. 

```env
# MongoDB Configuration
MONGODB_URI=mongodb://localhost:27017
DATABASE_NAME=judicio

# Qdrant Configuration
QDRANT_URL=http://localhost:6333
QDRANT_API_KEY=your_qdrant_api_key_here  # Leave empty if using local unauthenticated Qdrant
QDRANT_COLLECTION=judgements

# Application Settings
LOG_LEVEL=INFO

# Networking / Offline mode (Optional)
# Set HF_HUB_OFFLINE=1 AFTER the models have downloaded successfully the first time.
# HF_HUB_OFFLINE=1
```

---

## 🗄️ Data Initialization & Indexing

The project comes with a dedicated startup script (`app/setup.py`) to hydrate Qdrant with MongoDB data, generate embeddings, and build the local BM25 and Registry cache.

**To run the migration/setup:**
Ensure your virtual environment is active and run:
```bash
python -m app.setup
```
> **Note:** The first time you run this, it will download the BAAI models to the `llm-models/` folder. This may take some time depending on your internet connection. Generating embeddings is CPU-bound; it will safely pick up where it left off if interrupted.

**To monitor progress:**
You can open a secondary terminal tab, activate your virtual environment, and run the realtime watchdog script:
```bash
python -m scripts.watch
```

---

## 🏃 Running the Server

Once your initial data is seeded, you can start the FastAPI application server:

```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000 --reload
```

- **API Documentation (Swagger UI):** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 📂 Project Structure

```text
app/
 ├── main.py                # FastAPI app entry point & lifespan events
 ├── config/                # Database clients (Mongo, etc.)
 ├── core/                  # Global Pydantic configs & Logging
 ├── models/                # Core database models
 ├── modules/               # Feature-based modular logic
 │    ├── auth/             # User Authentication
 │    ├── chatbot/          # Chat functionality endpoints
 │    ├── prediction/       # ML logic for case predictions
 │    ├── retrieval/        # Hybrid Search (BM25 + Qdrant) & Reranking
 │    └── summarize/        # Legal document summarization
 └── schema/                # Pydantic validation schemas
llm-models/                 # Local cache for downloaded HuggingFace models
setup.py                    # Migration script (Mongo -> Qdrant vector embeddings)
watch.py                    # Realtime sync status viewer script
```

---

## 💡 Performance Tips

1. **Offline Mode:** Once the models are fully downloaded to the `llm-models` directory, add `HF_HUB_OFFLINE=1` to your `.env` file. This skips network checks at startup and boots the server significantly faster.
2. **Indexing:** Embedding legal documents is heavily computational. The `setup.py` script automatically skips documents it has already embedded. Let it run to completion for accurate retrieval results. 
