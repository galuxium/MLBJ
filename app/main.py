from __future__ import annotations

import asyncio
import os
from contextlib import asynccontextmanager

from dotenv import load_dotenv
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

load_dotenv()

from app.core.config import get_settings
from app.core.logging import configure_logging

settings = get_settings()
configure_logging(settings.log_level)

# Once both models are cached locally, set HF_HUB_OFFLINE=1 in your .env to
# skip all network calls at startup (~1-2 s faster). On first run, leave it
# unset so the models can download.
if os.getenv("HF_HUB_OFFLINE") == "1":
    os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")

import structlog
logger = structlog.get_logger()

from app.config.mongoClient import close_mongo_client, get_collection, get_database
from app.modules.auth.router import ensure_auth_indexes, router as auth_router
from app.modules.chatbot.router import router as chatbot_router
from app.modules.summarize.router import router as summarize_router
from app.modules.retrieval.router import router as retrieval_router

from app.modules.retrieval.services.embedder import EmbeddingService
from app.modules.retrieval.services.reranker import RerankerService
from app.modules.retrieval.services.sparse import SparseSearchService
from app.modules.retrieval.services.registry import DocumentRegistry
from app.modules.retrieval.services.search import SearchService
from app.modules.retrieval.services.indexing import IndexingService


@asynccontextmanager
async def lifespan(app: FastAPI):
    # ── Startup ───────────────────────────────────────────────────────────────
    logger.info("startup_begin")

    # MongoDB
    db = await get_database()
    app.state.db = db
    await ensure_auth_indexes()
    logger.info("mongodb_ready")

    # Qdrant — generous timeout because CPU embedding can take 30–60 s per batch
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )
    await _ensure_qdrant_collection(qdrant, settings)
    logger.info("qdrant_ready", url=settings.qdrant_url)

    # Inference models — loaded once per worker process
    embedder = EmbeddingService(
        model_name=settings.embedding_model,
        local_models_dir=settings.local_models_dir,
        device=settings.device,
        query_prefix=settings.bge_query_prefix,
    )
    reranker = RerankerService(
        model_name=settings.cross_encoder_model,
        device=settings.device,
        local_models_dir=settings.local_models_dir,
    )

    # Stateful services
    sparse   = SparseSearchService()
    registry = DocumentRegistry()

    judgments_collection = await get_collection("judgements")

    indexing_svc = IndexingService(
        embedder=embedder,
        sparse=sparse,
        registry=registry,
        qdrant=qdrant,
        mongo_collection=judgments_collection,
        settings=settings,
    )
    search_svc = SearchService(
        embedder=embedder,
        reranker=reranker,
        sparse=sparse,
        registry=registry,
        qdrant=qdrant,
        settings=settings,
    )

    # Populate registry + BM25 from MongoDB; sync Qdrant if stale.
    # Runs as a background task so the server accepts requests immediately.
    # Searches return partial results while indexing is in progress.
    asyncio.create_task(indexing_svc.load_from_mongo())

    # Attach to app.state so dependencies.py can retrieve them
    app.state.search_service    = search_svc
    app.state.indexing_service  = indexing_svc
    app.state.document_registry = registry

    logger.info("startup_complete", docs_loaded=registry.size())
    yield

    # ── Shutdown ──────────────────────────────────────────────────────────────
    embedder.shutdown()
    reranker.shutdown()
    await qdrant.close()
    await close_mongo_client()
    logger.info("shutdown_complete")


async def _ensure_qdrant_collection(qdrant: AsyncQdrantClient, cfg) -> None:
    """Create the Qdrant collection if it does not already exist."""
    try:
        await qdrant.get_collection(cfg.qdrant_collection)
    except (UnexpectedResponse, Exception):
        # Embedder isn't constructed yet so we hardcode the BGE-large dimension.
        # If you change the embedding model, update this value or read it from
        # a temporary model load.
        dim = int(os.getenv("QDRANT_VECTOR_DIM", "1024"))
        await qdrant.create_collection(
            collection_name=cfg.qdrant_collection,
            vectors_config=VectorParams(size=dim, distance=Distance.COSINE),
        )
        logger.info("qdrant_collection_created", collection=cfg.qdrant_collection, dim=dim)


app = FastAPI(
    title="Judicio AI",
    version="0.2.0",
    lifespan=lifespan,
    docs_url="/docs" if settings.env != "production" else None,
    redoc_url=None,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.cors_origins.split(","),
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/")
async def read_root():
    return JSONResponse(content={"message": "Judicio AI is running."}, status_code=200)


app.include_router(auth_router)
app.include_router(chatbot_router)
app.include_router(retrieval_router)
app.include_router(summarize_router)
