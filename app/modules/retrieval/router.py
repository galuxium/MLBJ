"""
Retrieval router — intentionally thin.

Route handlers contain zero business logic: they validate input, delegate
to injected services, and format the response. All retrieval logic lives
in services/search.py and services/indexing.py.
"""
from __future__ import annotations

import structlog
from fastapi import BackgroundTasks, Depends, HTTPException
from fastapi.routing import APIRouter

from .dependencies import get_indexing_service, get_registry, get_search_service
from .models import (
    IndexResponse,
    Judgment,
    JudgmentBatch,
    SearchDocQuery,
    SearchDocResponse,
    SearchQuery,
    SearchResponse,
)
from .services.indexing import IndexingService
from .services.registry import DocumentRegistry
from .services.search import SearchService

logger = structlog.get_logger()
router = APIRouter(prefix="/retrieval", tags=["retrieval"])


# ── Indexing ──────────────────────────────────────────────────────────────────

@router.post("/judgments/index", status_code=202, response_model=IndexResponse)
async def index_judgments(
    batch: JudgmentBatch,
    background_tasks: BackgroundTasks,
    svc: IndexingService = Depends(get_indexing_service),
) -> IndexResponse:
    """
    Accept a batch of judgments for indexing.

    Returns 202 Accepted immediately; indexing runs as a background task so
    the caller is never blocked by embedding or BM25 rebuild time.

    Error guarantee: errors are logged server-side (structured JSON logs).
    For guaranteed delivery with crash recovery, replace BackgroundTasks
    with an arq / Celery queue — only this handler needs to change.
    """
    background_tasks.add_task(svc.index_batch, batch.judgments)
    return IndexResponse(
        status="accepted",
        message=f"{len(batch.judgments)} judgment(s) queued for indexing.",
    )


@router.delete("/judgments/{case_no}")
async def delete_judgment(
    case_no: str,
    svc: IndexingService = Depends(get_indexing_service),
) -> dict:
    try:
        await svc.delete(case_no)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc))
    except RuntimeError as exc:
        raise HTTPException(status_code=500, detail=str(exc))
    return {"status": "success", "message": f"Judgment '{case_no}' deleted."}


# ── Search ────────────────────────────────────────────────────────────────────

@router.post("/search/hybrid/query", response_model=SearchResponse)
async def hybrid_search(
    search_query: SearchQuery,
    svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """
    Hybrid search with optional WordNet query expansion and cross-encoder reranking.

    Pipeline: expand → dense (Qdrant) ∥ sparse (BM25) → fuse → rerank (BGE).
    """
    return await svc.search(search_query)


@router.post("/search/hybrid/docs", response_model=SearchDocResponse)
async def hybrid_docs_search(
    search_query: SearchDocQuery,
    svc: SearchService = Depends(get_search_service),
) -> SearchDocResponse:
    """
    Hybrid search using an already-indexed judgment as the query document.
    Returns 404 if the case_no is not in the index.
    """
    return await svc.search_by_doc(search_query)


@router.post("/search/dense", response_model=SearchResponse)
async def dense_search_only(
    search_query: SearchQuery,
    svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Dense-only retrieval via Qdrant (no BM25, no reranking)."""
    return await svc.search_dense_only(search_query)


@router.post("/search/sparse", response_model=SearchResponse)
async def sparse_search_only(
    search_query: SearchQuery,
    svc: SearchService = Depends(get_search_service),
) -> SearchResponse:
    """Sparse-only retrieval via BM25 (no dense, no reranking)."""
    return await svc.search_sparse_only(search_query)


# ── Utility ───────────────────────────────────────────────────────────────────

@router.get("/judgments/{case_no}", response_model=Judgment)
async def get_judgment(
    case_no: str,
    registry: DocumentRegistry = Depends(get_registry),
    svc: SearchService = Depends(get_search_service),
) -> Judgment:
    if not registry.exists(case_no):
        raise HTTPException(status_code=404, detail="Judgment not found.")
    payload = await svc.get_by_case_no(case_no)
    if not payload:
        raise HTTPException(status_code=404, detail="Judgment payload not found in Qdrant.")
    return Judgment(**payload)


@router.get("/stats")
async def get_stats(
    registry: DocumentRegistry = Depends(get_registry),
    svc: SearchService = Depends(get_search_service),
) -> dict:
    return {"total_judgments": registry.size(), **svc.stats()}
