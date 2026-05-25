"""
DocumentRegistry — lightweight in-memory lookup layer.

Stores only the minimum data needed at request time:
  - combined text   (for cross-encoder reranking)
  - BM25 tokens     (for SparseSearchService rebuild)
  - mongo_id        (for MongoDB delete operations)

Full judgment payloads live in Qdrant and are fetched on demand.
This keeps the registry small and avoids duplicating large text blobs
three times (MongoDB + Qdrant + here).
"""
from __future__ import annotations

import asyncio
from typing import Any


class DocumentRegistry:
    def __init__(self) -> None:
        self._texts: dict[str, str] = {}           # case_no → combined_text
        self._tokens: dict[str, list[str]] = {}    # case_no → BM25 token list
        self._mongo_ids: dict[str, str] = {}       # case_no → MongoDB ObjectId str
        self._lock = asyncio.Lock()

    async def register(
        self,
        case_no: str,
        mongo_id: str,
        text: str,
        tokens: list[str],
    ) -> None:
        async with self._lock:
            self._texts[case_no] = text
            self._tokens[case_no] = tokens
            self._mongo_ids[case_no] = mongo_id

    async def deregister(self, case_no: str) -> None:
        async with self._lock:
            self._texts.pop(case_no, None)
            self._tokens.pop(case_no, None)
            self._mongo_ids.pop(case_no, None)

    def get_text(self, case_no: str) -> str:
        return self._texts.get(case_no, "")

    def get_texts(self, case_nos: list[str]) -> list[str]:
        return [self._texts.get(cn, "") for cn in case_nos]

    def get_mongo_id(self, case_no: str) -> str | None:
        return self._mongo_ids.get(case_no)

    def exists(self, case_no: str) -> bool:
        return case_no in self._texts

    def token_corpus(self) -> dict[str, list[str]]:
        """Return a snapshot {case_no: tokens} for BM25 rebuild."""
        return dict(self._tokens)

    def size(self) -> int:
        return len(self._texts)

    def case_nos(self) -> list[str]:
        return list(self._texts.keys())

    def dump_state(self) -> dict:
        return {
            "texts": dict(self._texts),
            "tokens": dict(self._tokens),
            "mongo_ids": dict(self._mongo_ids),
        }

    def load_state(self, state: dict) -> None:
        self._texts = dict(state["texts"])
        self._tokens = dict(state["tokens"])
        self._mongo_ids = dict(state["mongo_ids"])
