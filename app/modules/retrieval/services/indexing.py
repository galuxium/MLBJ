"""
IndexingService — owns all index mutations.

Responsibilities:
  1. Persist documents to MongoDB (source of truth)
  2. Embed documents and upsert to Qdrant (dense store)
  3. Register in DocumentRegistry (text + tokens for reranking / BM25)
  4. Trigger BM25 rebuild in SparseSearchService

Background-task pattern:
  The router returns 202 Accepted immediately and calls `index_batch` via
  FastAPI BackgroundTasks. This decouples indexing latency from the HTTP
  response cycle.

  For guaranteed delivery / crash recovery, replace BackgroundTasks with
  an arq or Celery task queue. The IndexingService methods are already
  async and queue-agnostic — only the router wiring changes.
"""
from __future__ import annotations

import asyncio
import hashlib
import pickle
from pathlib import Path

import structlog
from pymongo.errors import PyMongoError
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import ResponseHandlingException, UnexpectedResponse
from qdrant_client.models import PointStruct

from app.core.config import Settings
from app.modules.retrieval.models import Judgment
from app.modules.retrieval.utils import combine_text, stable_point_id
from .embedder import EmbeddingService
from .registry import DocumentRegistry
from .sparse import SparseSearchService

logger = structlog.get_logger()

_BATCH_SIZE = 32  # documents per embedding batch at startup
_CACHE_FILENAME = "index_cache.pkl"


class IndexingService:
    def __init__(
        self,
        embedder: EmbeddingService,
        sparse: SparseSearchService,
        registry: DocumentRegistry,
        qdrant: AsyncQdrantClient,
        mongo_collection,
        settings: Settings,
    ) -> None:
        self._embedder = embedder
        self._sparse = sparse
        self._registry = registry
        self._qdrant = qdrant
        self._mongo = mongo_collection
        self._settings = settings

    # ── Public API ────────────────────────────────────────────────────────────

    async def index_batch(self, judgments: list[Judgment]) -> None:
        """
        Upsert a batch of judgments into all stores.
        Designed to run as a background task — errors are logged, not raised.
        """
        log = logger.bind(batch_size=len(judgments))
        try:
            texts = [combine_text(j.model_dump()) for j in judgments]
            embeddings = await self._embedder.encode_documents(texts)

            points: list[PointStruct] = []
            for judgment, text, emb in zip(judgments, texts, embeddings):
                data = judgment.model_dump()
                mongo_id = await self._upsert_mongo(judgment.case_no, data)
                tokens = self._sparse.tokenize(text)

                points.append(
                    PointStruct(
                        id=stable_point_id(judgment.case_no),
                        vector=emb.tolist(),
                        payload=data,
                    )
                )
                await self._registry.register(judgment.case_no, mongo_id, text, tokens)

            await self._qdrant_upsert(points)
            await self._sparse.rebuild(self._registry.token_corpus())
            self._refresh_cache()

            log.info("batch_indexed", total_docs=self._registry.size())

        except (PyMongoError, UnexpectedResponse, RuntimeError) as exc:
            log.error("index_batch_failed", error=str(exc), exc_info=True)

    async def delete(self, case_no: str) -> None:
        """Delete a judgment from all stores. Raises ValueError if not found."""
        if not self._registry.exists(case_no):
            raise ValueError(f"Case '{case_no}' not found in index.")

        mongo_id = self._registry.get_mongo_id(case_no)
        log = logger.bind(case_no=case_no)
        try:
            if mongo_id:
                from bson import ObjectId
                await self._mongo.delete_one({"_id": ObjectId(mongo_id)})

            from qdrant_client.models import PointIdsList
            await self._qdrant.delete(
                collection_name=self._settings.qdrant_collection,
                points_selector=PointIdsList(points=[stable_point_id(case_no)]),
            )
            await self._registry.deregister(case_no)
            await self._sparse.rebuild(self._registry.token_corpus())
            self._refresh_cache()
            log.info("judgment_deleted", total_docs=self._registry.size())

        except (PyMongoError, UnexpectedResponse) as exc:
            log.error("delete_failed", error=str(exc), exc_info=True)
            raise RuntimeError(f"Delete failed: {exc}") from exc

    async def load_from_mongo(self) -> None:
        """
        Startup loader — hydrate DocumentRegistry + BM25 from MongoDB,
        and upsert to Qdrant any documents not already there.

        Fast path: if a valid pickled cache exists on disk AND Qdrant is in sync,
        load registry + BM25 directly (~2 s for 5k docs vs ~45 s cold rebuild).

        Slow path: stream from MongoDB, tokenize, optionally re-embed, then
        write a fresh cache to disk for next startup.
        """
        log = logger.bind(collection=self._settings.qdrant_collection)
        log.info("startup_load_begin")

        mongo_count = await self._mongo.count_documents({})
        qdrant_count = await self._qdrant_count()
        case_nos = await self._fetch_mongo_case_nos()
        signature = _compute_signature(case_nos)

        if qdrant_count == mongo_count and self._try_load_cache(signature):
            log.info("startup_load_from_cache", total_docs=self._registry.size())
            return

        needs_embed = qdrant_count != mongo_count
        if needs_embed:
            log.info(
                "qdrant_stale",
                mongo_count=mongo_count,
                qdrant_count=qdrant_count,
                action="re_embedding",
            )

        batch: list[dict] = []
        async for doc in self._mongo.find():
            batch.append(doc)
            if len(batch) >= _BATCH_SIZE:
                try:
                    await self._flush_startup_batch(batch, embed=needs_embed)
                except (ResponseHandlingException, UnexpectedResponse, Exception) as exc:
                    log.error("startup_batch_failed", error=str(exc), docs_so_far=self._registry.size())
                batch = []
        if batch:
            try:
                await self._flush_startup_batch(batch, embed=needs_embed)
            except (ResponseHandlingException, UnexpectedResponse, Exception) as exc:
                log.error("startup_batch_failed", error=str(exc), docs_so_far=self._registry.size())

        await self._sparse.rebuild(self._registry.token_corpus())
        self._save_cache(signature)
        log.info("startup_load_complete", total_docs=self._registry.size())

    # ── Internal helpers ──────────────────────────────────────────────────────

    async def _flush_startup_batch(self, docs: list[dict], embed: bool) -> None:
        texts = [combine_text(doc) for doc in docs]

        if embed:
            # Check which documents already exist in Qdrant to avoid re-embedding
            target_ids = [stable_point_id(doc["case_no"]) for doc in docs]
            existing_records = []
            try:
                existing_records = await self._qdrant.retrieve(
                    collection_name=self._settings.qdrant_collection,
                    ids=target_ids,
                    with_payload=False,
                    with_vectors=False,
                )
            except Exception:
                pass
            
            existing_ids = {record.id for record in existing_records}
            
            # Filter to only the documents (and their texts) that need embedding
            missing_indices = [i for i, doc in enumerate(docs) if stable_point_id(doc["case_no"]) not in existing_ids]
            
            if missing_indices:
                missing_texts = [texts[i] for i in missing_indices]
                missing_docs = [docs[i] for i in missing_indices]
                
                embeddings = await self._embedder.encode_documents(missing_texts)
                points = [
                    PointStruct(
                        id=stable_point_id(doc["case_no"]),
                        vector=emb.tolist(),
                        payload={k: v for k, v in doc.items() if k != "_id"},
                    )
                    for doc, emb in zip(missing_docs, embeddings)
                ]
                await self._qdrant_upsert(points)

        coros = [
            self._registry.register(
                case_no=doc["case_no"],
                mongo_id=str(doc["_id"]),
                text=text,
                tokens=self._sparse.tokenize(text),
            )
            for doc, text in zip(docs, texts)
        ]
        await asyncio.gather(*coros)

    async def _upsert_mongo(self, case_no: str, data: dict) -> str:
        existing = await self._mongo.find_one({"case_no": case_no}, {"_id": 1})
        if existing:
            # Skip replacing if the case already exists
            return str(existing["_id"])
        result = await self._mongo.insert_one(data)
        return str(result.inserted_id)

    async def _qdrant_upsert(self, points: list[PointStruct]) -> None:
        if not points:
            return
        try:
            await self._qdrant.upsert(
                collection_name=self._settings.qdrant_collection,
                points=points,
            )
        except ResponseHandlingException as exc:
            logger.error("qdrant_upsert_timeout", error=str(exc))
            raise
        except UnexpectedResponse as exc:
            logger.error("qdrant_upsert_failed", status=exc.status_code, reason=exc.reason_phrase)
            raise

    async def _qdrant_count(self) -> int:
        try:
            info = await self._qdrant.get_collection(self._settings.qdrant_collection)
            return info.points_count or 0
        except UnexpectedResponse:
            return 0

    async def _fetch_mongo_case_nos(self) -> list[str]:
        cursor = self._mongo.find({}, {"case_no": 1, "_id": 0})
        return [doc["case_no"] async for doc in cursor]

    # ── Disk cache for registry + BM25 ────────────────────────────────────────

    @property
    def _cache_path(self) -> Path:
        return self._settings.local_models_dir / _CACHE_FILENAME

    def _try_load_cache(self, signature: str) -> bool:
        path = self._cache_path
        if not path.exists():
            return False
        try:
            with open(path, "rb") as f:
                data = pickle.load(f)
        except Exception as exc:
            logger.warning("cache_load_failed", error=str(exc))
            return False
        if data.get("signature") != signature:
            logger.info("cache_stale", action="rebuild")
            return False
        try:
            self._registry.load_state(data["registry"])
            self._sparse.load_state(data["sparse"])
        except (KeyError, TypeError) as exc:
            logger.warning("cache_schema_mismatch", error=str(exc))
            return False
        return True

    def _save_cache(self, signature: str) -> None:
        path = self._cache_path
        path.parent.mkdir(parents=True, exist_ok=True)
        data = {
            "signature": signature,
            "registry": self._registry.dump_state(),
            "sparse": self._sparse.dump_state(),
        }
        tmp = path.with_suffix(".pkl.tmp")
        with open(tmp, "wb") as f:
            pickle.dump(data, f, protocol=pickle.HIGHEST_PROTOCOL)
        tmp.replace(path)  # atomic on POSIX & modern Windows
        logger.info("cache_saved", path=str(path), num_docs=self._registry.size())

    def _refresh_cache(self) -> None:
        """Write a fresh cache after a successful mutation."""
        try:
            signature = _compute_signature(self._registry.case_nos())
            self._save_cache(signature)
        except Exception as exc:
            logger.warning("cache_refresh_failed", error=str(exc))


def _compute_signature(case_nos: list[str]) -> str:
    h = hashlib.sha256()
    for cn in sorted(case_nos):
        h.update(cn.encode("utf-8"))
        h.update(b"\x00")
    return h.hexdigest()
