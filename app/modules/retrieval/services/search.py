"""
SearchService — orchestrates the full retrieval pipeline.

Pipeline (per request):
  1. Optional WordNet query expansion (retrieval only)
  2. Dense (Qdrant cosine) + Sparse (BM25) retrieval in parallel
  3. Min-max weighted fusion → candidate shortlist
  4. Optional cross-encoder reranking on shortlist
  5. Qdrant payload hydration → final SearchResult list

Depends on:
  EmbeddingService, RerankerService, SparseSearchService,
  DocumentRegistry, AsyncQdrantClient, Settings
"""
from __future__ import annotations

import asyncio
import time
from typing import Any

import structlog
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

from app.core.config import Settings
from app.modules.retrieval.models import (
    FusionStrategy,
    SearchDocQuery,
    SearchDocResponse,
    SearchQuery,
    SearchResponse,
    SearchResult,
)
from app.modules.retrieval.utils import (
    build_semantic_representation,
    expand_query,
    reciprocal_rank_fusion,
    stable_point_id,
    weighted_fusion,
)
from .embedder import EmbeddingService
from .reranker import RerankerService
from .registry import DocumentRegistry
from .sparse import SparseSearchService

logger = structlog.get_logger()


class SearchService:
    def __init__(
        self,
        embedder: EmbeddingService,
        reranker: RerankerService,
        sparse: SparseSearchService,
        registry: DocumentRegistry,
        qdrant: AsyncQdrantClient,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._reranker = reranker
        self._sparse = sparse
        self._registry = registry
        self._qdrant = qdrant
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    async def search(self, request: SearchQuery) -> SearchResponse:
        t0 = time.perf_counter()
        log = logger.bind(query=request.query, top_k=request.top_k)

        shortlist_n = request.top_k * self._settings.shortlist_multiplier
        retrieval_query = (
            expand_query(
                request.query,
                self._settings.max_syns_per_token,
                self._settings.max_hypos_per_token,
            )
            if request.expand_query and self._settings.use_query_expansion
            else request.query
        )

        dense_scores, sparse_scores = await asyncio.gather(
            self._dense_search(retrieval_query, shortlist_n),
            self._sparse.search(retrieval_query, shortlist_n),
        )

        fused = self._fuse(dense_scores, sparse_scores, request.fusion, request.dense_weight)
        shortlisted = sorted(fused, key=lambda x: fused[x], reverse=True)[:shortlist_n]

        if request.enable_rerank and shortlisted:
            texts = self._registry.get_texts(shortlisted)
            reranked = await self._reranker.rerank(request.query, shortlisted, texts, request.top_k)
            final_case_nos = [r[0] for r in reranked]
            rerank_meta = {r[0]: (r[1], r[2]) for r in reranked}
        else:
            final_case_nos = shortlisted[: request.top_k]
            rerank_meta = {}

        payloads = await self._fetch_payloads(final_case_nos)
        results = [
            self._to_result(
                payload,
                dense_scores,
                sparse_scores,
                fused,
                rerank_meta,
            )
            for payload in payloads
            if payload
        ]

        log.info(
            "search_completed",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            num_results=len(results),
            reranked=bool(rerank_meta),
        )
        return SearchResponse(results=results, query=request.query, total_results=len(results))

    async def search_by_doc(self, request: SearchDocQuery) -> SearchDocResponse:
        t0 = time.perf_counter()
        case_no = request.query.case_no
        log = logger.bind(case_no=case_no, top_k=request.top_k)

        if not self._registry.exists(case_no):
            from fastapi import HTTPException
            raise HTTPException(
                status_code=404,
                detail=f"Case '{case_no}' is not indexed. POST to /retrieval/judgments/index first.",
            )

        shortlist_n = request.top_k * self._settings.shortlist_multiplier
        query_text = request.query.get_combined_text()

        dense_scores, sparse_scores = await asyncio.gather(
            self._dense_search(query_text, shortlist_n),
            self._sparse.search(query_text, shortlist_n),
        )

        fused = weighted_fusion(dense_scores, sparse_scores, request.dense_weight)
        shortlisted = sorted(fused, key=lambda x: fused[x], reverse=True)[:shortlist_n]

        if request.enable_rerank and shortlisted:
            texts = self._registry.get_texts(shortlisted)
            reranked = await self._reranker.rerank(query_text, shortlisted, texts, request.top_k)
            final_case_nos = [r[0] for r in reranked]
            rerank_meta = {r[0]: (r[1], r[2]) for r in reranked}
        else:
            final_case_nos = shortlisted[: request.top_k]
            rerank_meta = {}

        payloads = await self._fetch_payloads(final_case_nos)
        results = [
            self._to_result(payload, dense_scores, sparse_scores, fused, rerank_meta)
            for payload in payloads
            if payload
        ]

        log.info(
            "doc_search_completed",
            latency_ms=round((time.perf_counter() - t0) * 1000, 1),
            num_results=len(results),
        )
        return SearchDocResponse(
            results=results,
            query=request.query,
            total_results=len(results),
            semantic_representation=build_semantic_representation(
                [p for p in payloads if p]
            ),
        )

    async def search_dense_only(self, request: SearchQuery) -> SearchResponse:
        t0 = time.perf_counter()
        dense_scores = await self._dense_search(request.query, request.top_k)
        ranked = sorted(dense_scores, key=lambda x: dense_scores[x], reverse=True)[: request.top_k]
        payloads = await self._fetch_payloads(ranked)
        results = [self._to_result(p, dense_scores, {}, dense_scores, {}) for p in payloads if p]
        logger.info("dense_search_completed", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return SearchResponse(results=results, query=request.query, total_results=len(results))

    async def search_sparse_only(self, request: SearchQuery) -> SearchResponse:
        t0 = time.perf_counter()
        sparse_scores = await self._sparse.search(request.query, request.top_k)
        ranked = sorted(sparse_scores, key=lambda x: sparse_scores[x], reverse=True)
        payloads = await self._fetch_payloads(ranked)
        results = [self._to_result(p, {}, sparse_scores, sparse_scores, {}) for p in payloads if p]
        logger.info("sparse_search_completed", latency_ms=round((time.perf_counter() - t0) * 1000, 1))
        return SearchResponse(results=results, query=request.query, total_results=len(results))

    async def get_by_case_no(self, case_no: str) -> dict | None:
        payloads = await self._fetch_payloads([case_no])
        return payloads[0] if payloads else None

    def stats(self) -> dict:
        return {
            "embedding_model": self._settings.embedding_model,
            "cross_encoder_model": self._settings.cross_encoder_model,
            "embedding_dimension": self._embedder.dimension,
        }

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _dense_search(self, query: str, top_k: int) -> dict[str, float]:
        q_emb = await self._embedder.encode_query(query)
        try:
            response = await self._qdrant.query_points(
                collection_name=self._settings.qdrant_collection,
                query=q_emb[0].tolist(),
                limit=top_k,
                with_payload=["case_no"],
            )
        except UnexpectedResponse as exc:
            logger.error("qdrant_search_failed", status=exc.status_code, reason=exc.reason_phrase)
            return {}
        return {hit.payload["case_no"]: hit.score for hit in response.points if hit.payload}

    def _fuse(
        self,
        dense: dict[str, float],
        sparse: dict[str, float],
        strategy: FusionStrategy,
        dense_weight: float,
    ) -> dict[str, float]:
        if strategy == FusionStrategy.RRF:
            return reciprocal_rank_fusion(dense, sparse)
        return weighted_fusion(dense, sparse, dense_weight)

    async def _fetch_payloads(self, case_nos: list[str]) -> list[dict[str, Any]]:
        """Batch-retrieve full judgment payloads from Qdrant by point ID."""
        if not case_nos:
            return []
        point_ids = [stable_point_id(cn) for cn in case_nos]
        try:
            points = await self._qdrant.retrieve(
                collection_name=self._settings.qdrant_collection,
                ids=point_ids,
                with_payload=True,
            )
        except UnexpectedResponse as exc:
            logger.error("qdrant_retrieve_failed", status=exc.status_code)
            return []
        by_case = {p.payload["case_no"]: p.payload for p in points if p.payload}
        # preserve the order of case_nos (reranker order)
        return [by_case.get(cn, {}) for cn in case_nos]

    @staticmethod
    def _to_result(
        payload: dict[str, Any],
        dense_scores: dict[str, float],
        sparse_scores: dict[str, float],
        fused: dict[str, float],
        rerank_meta: dict[str, tuple[float, float]],
    ) -> SearchResult:
        cn = payload.get("case_no", "")
        logit, prob = rerank_meta.get(cn, (None, None))
        return SearchResult(
            case_no=cn,
            title=payload.get("title", ""),
            jurisdiction=payload.get("jurisdiction"),
            date=payload.get("date"),
            issue=payload.get("issue"),
            facts=payload.get("facts"),
            court_reasoning=payload.get("court_reasoning"),
            precedent_analysis=payload.get("precedent_analysis"),
            argument_by_petitioner=payload.get("argument_by_petitioner"),
            conclusion=payload.get("conclusion"),
            ipc_sections=payload.get("ipc_sections"),
            statute_analysis=payload.get("statute_analysis"),
            argument_by_respondent=payload.get("argument_by_respondent"),
            dense_score=dense_scores.get(cn, 0.0),
            sparse_score=sparse_scores.get(cn, 0.0),
            combined_score=fused.get(cn, 0.0),
            rerank_score=logit,
            rerank_prob=prob,
        )
