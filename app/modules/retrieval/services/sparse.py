"""
SparseSearchService — BM25 in-memory keyword retrieval.

Stable interface contract (swap for Elasticsearch / MongoDB Atlas Search
without touching SearchService or the router):

    async def search(query: str, top_k: int) -> dict[str, float]
    async def rebuild(token_corpus: dict[str, list[str]]) -> None

To switch to Elasticsearch:
  1. Create an ElasticsearchSparseService that implements the same two methods.
  2. Swap it in the lifespan wiring in main.py.
  3. No other file needs to change.
"""
from __future__ import annotations

import asyncio
import re
from typing import Optional

import numpy as np
import structlog
from nltk.stem import PorterStemmer
from rank_bm25 import BM25Okapi

from app.modules.retrieval.utils import STOPWORDS

logger = structlog.get_logger()
_WORD_RE = re.compile(r"[a-z]+")


class SparseSearchService:
    def __init__(self) -> None:
        self._stemmer = PorterStemmer()
        self._index: Optional[BM25Okapi] = None
        self._case_nos: list[str] = []
        self._lock = asyncio.Lock()

    def tokenize(self, text: str) -> list[str]:
        """Porter-stemmed, NLTK-stopword-filtered tokenization."""
        return [
            self._stemmer.stem(tok)
            for tok in _WORD_RE.findall(text.lower())
            if tok not in STOPWORDS and len(tok) > 1
        ]

    async def rebuild(self, token_corpus: dict[str, list[str]]) -> None:
        """
        Atomically replace the BM25 index from a {case_no: tokens} snapshot.
        Called at startup and after every index mutation.
        """
        if not token_corpus:
            async with self._lock:
                self._index = None
                self._case_nos = []
            return

        case_nos = list(token_corpus.keys())
        new_index = BM25Okapi([token_corpus[cn] for cn in case_nos])

        async with self._lock:
            self._index = new_index
            self._case_nos = case_nos

        logger.info("sparse_index_rebuilt", num_docs=len(case_nos))

    def dump_state(self) -> dict:
        return {"bm25": self._index, "case_nos": list(self._case_nos)}

    def load_state(self, state: dict) -> None:
        self._index = state["bm25"]
        self._case_nos = list(state["case_nos"])

    async def search(self, query: str, top_k: int) -> dict[str, float]:
        async with self._lock:
            index = self._index
            case_nos = self._case_nos[:]

        if index is None or not case_nos:
            return {}

        tokens = self.tokenize(query)
        scores = index.get_scores(tokens)
        top_indices = np.argsort(scores)[::-1][:top_k]
        return {
            case_nos[i]: float(scores[i])
            for i in top_indices
            if scores[i] > 0
        }
