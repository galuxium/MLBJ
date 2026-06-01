# Judicio AI Backend - API Architecture & Endpoints

This document outlines the API endpoints, their request/response formats, and examples for the Judicio AI backend. The application is built using FastAPI and organizes its functionality into distinct modules.

---

## Global Endpoints

### 1. Root Check
- **Endpoint**: `GET /`
- **Description**: Verifies if the backend server is running.
- **Request**: None
- **Response**:
  ```json
  {
    "message": "Judicio AI is running."
  }
  ```

---

## Authentication Module (`/auth`)

### 1. Health Check
- **Endpoint**: `GET /auth/health`
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### 2. User Registration
- **Endpoint**: `POST /auth/register`
- **Description**: Registers a new user.
- **Request Body**:
  ```json
  {
    "username": "john_doe",
    "email": "john.doe@example.com",
    "password": "securepassword123"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Registration successful"
  }
  ```

### 3. User Login
- **Endpoint**: `POST /auth/login`
- **Description**: Logs in a user and returns an access token.
- **Request Body**:
  ```json
  {
    "email": "john.doe@example.com",
    "password": "securepassword123"
  }
  ```
- **Response**:
  ```json
  {
    "message": "Login successful",
    "access_token": "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9..."
  }
  ```

---

## Chatbot Module (`/chatbot`)

### 1. Health Check
- **Endpoint**: `GET /chatbot/health`
- **Response**:
  ```json
  {
    "status": "healthy"
  }
  ```

### 2. Ask Question
- **Endpoint**: `POST /chatbot/ask`
- **Description**: General chatbot query endpoint.
- **Request Body**: Handled dynamically or without strict typing currently.
- **Response**:
  ```json
  {
    "message": "Question received"
  }
  ```

---

## Prediction Module (`/predict`)

### 1. Health Check
- **Endpoint**: `GET /predict/health`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Predict from Text
- **Endpoint**: `POST /predict/text`
- **Description**: Analyzes input text and predicts judgment outcomes.
- **Request Body**:
  ```json
  {
    "text": "This is a sample case description involving breach of contract..."
  }
  ```
- **Response**:
  ```json
  {
    "prediction": 1,
    "label": "Granted",
    "confidence": 0.85,
    "logits": [2.3, 5.1],
    "probabilities": [0.15, 0.85],
    "n_chunks_used": 1
  }
  ```

### 3. Predict from File
- **Endpoint**: `POST /predict/file`
- **Description**: Upload a text file for prediction.
- **Content-Type**: `multipart/form-data` or raw text bytes depending on client.
- **Response**: Extracted text undergoes same prediction processing as `/predict/text`, same response format.

---

## Retrieval Module (`/retrieval`)

### 1. Index Judgments
- **Endpoint**: `POST /retrieval/judgments/index`
- **Description**: Queues a batch of judgments for asynchronous indexing into Qdrant & MongoDB.
- **Request Body**:
  ```json
  {
    "judgments": [
      {
        "case_no": "SC-2023-01",
        "title": "State v. John Doe",
        "jurisdiction": "Supreme Court",
        "date": "2023-05-15",
        "issue": "Whether the contract was breached...",
        "facts": "The defendant failed to deliver the goods..."
      }
    ]
  }
  ```
- **Response**:
  ```json
  {
    "status": "accepted",
    "message": "1 judgment(s) queued for indexing."
  }
  ```

### 2. Delete Judgment
- **Endpoint**: `DELETE /retrieval/judgments/{case_no}`
- **Description**: Removes a specific judgment from the registry and index.
- **Response**:
  ```json
  {
    "status": "success",
    "message": "Judgment 'SC-2023-01' deleted."
  }
  ```

### 3. Get Specific Judgment
- **Endpoint**: `GET /retrieval/judgments/{case_no}`
- **Response**:
  Yields the JSON representation of the Judgment schema structure as indexed.

### 4. Hybrid Search Query
- **Endpoint**: `POST /retrieval/search/hybrid/query`
- **Description**: Performs a hybrid (dense + sparse) search with reranking.
- **Request Body**:
  ```json
  {
    "query": "Breach of contract concerning delivery of goods",
    "top_k": 10,
    "dense_weight": 0.6,
    "fusion": "weighted",
    "enable_rerank": true,
    "expand_query": true
  }
  ```
- **Response**:
  ```json
  {
    "query": "Breach of contract concerning delivery of goods",
    "total_results": 1,
    "results": [
      {
        "case_no": "SC-2023-01",
        "title": "State v. John Doe",
        "dense_score": 0.88,
        "sparse_score": 12.5,
        "combined_score": 0.92,
        "rerank_score": 4.5,
        "rerank_prob": 0.98
      }
    ]
  }
  ```

### 5. Document-Based Hybrid Search 
- **Endpoint**: `POST /retrieval/search/hybrid/docs`
- **Description**: Allows passing an entire document object as the query entity.

### 6. Search Stats
- **Endpoint**: `GET /retrieval/stats`
- **Response**: Provides metadata regarding overall index sizes and document count.

---

## Summarization Module (`/summarize`)
*Note: These endpoints require Authorization headers (Bearer Token) generated from the `/auth/login` endpoint.*

### 1. Health Check
- **Endpoint**: `GET /summarize/health`
- **Response**:
  ```json
  {
    "status": "ok"
  }
  ```

### 2. Summarize Text
- **Endpoint**: `POST /summarize/text`
- **Description**: Uses Gemini AI to extract structured case information from raw input text.
- **Request Body**:
  ```json
  {
    "query": "Detailed notes about a recent court case concerning industrial negligence..."
  }
  ```
- **Response**:
  ```json
  {
    "result": {
      "Facts": ["Point 1 about facts", "Point 2"],
      "Issues": ["Key issue identified"]
    },
    "timestamp": "2026-05-31T12:00:00Z"
  }
  ```

### 3. Extract From Uploaded File
- **Endpoint**: `POST /summarize/file`
- **Description**: Upload a PDF file to extract the raw text content for summarization. 
- **Content-Type**: `multipart/form-data` with `file` field (.pdf).
- **Response**:
  ```json
  {
    "data": "Extracted text content from the document...",
    "file": "case_document.pdf",
    "timestamp": "2026-05-31T12:00:00Z"
  }
  ```
