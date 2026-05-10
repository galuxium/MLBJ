"""
Colab GPU bulk-embed script.

Reads all judgments from MongoDB, embeds with BAAI/bge-large-en-v1.5 on GPU,
and upserts them to your Qdrant Cloud collection.

USAGE
─────
1. Open https://colab.research.google.com/ → New notebook
2. Runtime → Change runtime type → T4 GPU
3. Paste this entire file into a single cell
4. Fill in the four credentials in the CONFIG block below
5. Run the cell

The script is IDEMPOTENT — Qdrant upserts use deterministic UUIDs derived from
case_no, so re-running just overwrites existing points. Safe to interrupt and
restart. Once done, your local FastAPI server will see qdrant_count == mongo_count
and skip re-embedding entirely on next startup.

WHY GPU not TPU
───────────────
sentence-transformers is PyTorch-native; TPU/XLA requires extra plumbing and is
often slower for inference workloads of this size. T4 GPU (free Colab tier)
embeds 5022 docs in ~2-3 minutes vs hours on CPU.
"""

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 1 — install dependencies                                           ║
# ╚══════════════════════════════════════════════════════════════════════════╝
# !pip install -q sentence-transformers qdrant-client pymongo[srv] tqdm

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 2 — imports + config                                               ║
# ╚══════════════════════════════════════════════════════════════════════════╝
import time
import uuid
from typing import Any

import torch
from pymongo import MongoClient
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
from sentence_transformers import SentenceTransformer
from tqdm.auto import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────
# Paste your secrets here. Do NOT commit this file with secrets filled in.
MONGODB_URI       = "mongodb+srv://USER:PASS@cluster.mongodb.net/..."
DATABASE_NAME     = "judicio"
MONGO_COLLECTION  = "judgements"

QDRANT_URL        = "https://YOUR-CLUSTER.eu-west-1-0.aws.cloud.qdrant.io:6333"
QDRANT_API_KEY    = "YOUR_QDRANT_API_KEY"
QDRANT_COLLECTION = "judgements"

EMBEDDING_MODEL   = "BAAI/bge-large-en-v1.5"
EMBED_BATCH_SIZE  = 64    # GPU batch — raise to 128 if you have memory headroom
UPSERT_BATCH_SIZE = 256   # Qdrant points per upsert call

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 3 — verify GPU                                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"device = {device}")
if device == "cuda":
    print(f"gpu    = {torch.cuda.get_device_name(0)}")
else:
    print("WARNING: no GPU detected — switch runtime to T4 GPU before continuing")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 4 — helpers (mirror backend's utils.py exactly)                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def combine_text(data: dict[str, Any]) -> str:
    """Must match app/modules/retrieval/utils.py::combine_text exactly."""
    fields = [
        data.get("title", ""), data.get("issue"), data.get("facts"),
        data.get("court_reasoning"), data.get("precedent_analysis"),
        data.get("argument_by_petitioner"), data.get("conclusion"),
        data.get("statute_analysis"), data.get("argument_by_respondent"),
    ]
    parts: list[str] = []
    for f in fields:
        if f is None:
            continue
        if isinstance(f, list):
            parts.extend(str(i) for i in f if i)
        else:
            parts.append(str(f))
    return " ".join(parts)


def stable_point_id(case_no: str) -> str:
    """Must match app/modules/retrieval/utils.py::stable_point_id exactly."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, case_no))


# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 5 — pull docs from MongoDB                                         ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("connecting to mongo...")
mongo = MongoClient(MONGODB_URI)
coll = mongo[DATABASE_NAME][MONGO_COLLECTION]

mongo_count = coll.count_documents({})
print(f"mongo docs = {mongo_count}")

print("loading docs into memory...")
docs = list(coll.find({}))
print(f"loaded {len(docs)} docs")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 6 — load embedding model on GPU                                    ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print(f"loading {EMBEDDING_MODEL} on {device}...")
model = SentenceTransformer(EMBEDDING_MODEL, device=device)
dim = model.get_sentence_embedding_dimension()
print(f"embedding dimension = {dim}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 7 — ensure Qdrant collection exists                                ║
# ╚══════════════════════════════════════════════════════════════════════════╝
qdrant = QdrantClient(url=QDRANT_URL, api_key=QDRANT_API_KEY, timeout=120)

existing = [c.name for c in qdrant.get_collections().collections]
if QDRANT_COLLECTION not in existing:
    print(f"creating collection {QDRANT_COLLECTION!r} (dim={dim})")
    qdrant.create_collection(
        collection_name=QDRANT_COLLECTION,
        vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
    )
else:
    info = qdrant.get_collection(QDRANT_COLLECTION)
    print(f"collection exists — current points = {info.points_count}")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 8 — combine text + embed in GPU batches                            ║
# ╚══════════════════════════════════════════════════════════════════════════╝
print("building texts...")
texts = [combine_text(d) for d in docs]

print(f"embedding {len(texts)} docs in batches of {EMBED_BATCH_SIZE}...")
t0 = time.time()
embeddings = model.encode(
    texts,
    batch_size=EMBED_BATCH_SIZE,
    normalize_embeddings=True,
    show_progress_bar=True,
    convert_to_numpy=True,
)
elapsed = time.time() - t0
print(f"embedded {len(embeddings)} docs in {elapsed:.1f}s "
      f"({len(embeddings)/elapsed:.1f} docs/s)")

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 9 — upsert to Qdrant in chunks                                     ║
# ╚══════════════════════════════════════════════════════════════════════════╝
def to_point(doc: dict, embedding) -> PointStruct:
    payload = {k: v for k, v in doc.items() if k != "_id"}
    return PointStruct(
        id=stable_point_id(doc["case_no"]),
        vector=embedding.tolist(),
        payload=payload,
    )


print(f"upserting to qdrant in batches of {UPSERT_BATCH_SIZE}...")
total = len(docs)
for start in tqdm(range(0, total, UPSERT_BATCH_SIZE)):
    end = min(start + UPSERT_BATCH_SIZE, total)
    chunk = [to_point(docs[i], embeddings[i]) for i in range(start, end)]

    for attempt in range(3):
        try:
            qdrant.upsert(collection_name=QDRANT_COLLECTION, points=chunk, wait=True)
            break
        except Exception as exc:
            print(f"  upsert {start}-{end} failed (attempt {attempt+1}/3): {exc}")
            if attempt == 2:
                raise
            time.sleep(2 ** attempt)

# ╔══════════════════════════════════════════════════════════════════════════╗
# ║  CELL 10 — verify                                                        ║
# ╚══════════════════════════════════════════════════════════════════════════╝
info = qdrant.get_collection(QDRANT_COLLECTION)
print(f"\nDONE — qdrant points = {info.points_count} (expected {mongo_count})")
if info.points_count == mongo_count:
    print("✓ counts match — your local FastAPI server will skip re-embedding next startup")
else:
    print("✗ counts differ — investigate before relying on the index")
