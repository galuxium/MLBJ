"""
EmbeddingService — wraps SentenceTransformer for async use.

One instance is created at startup and reused for all requests, so the model
is loaded once per worker process (not per request).

To point this at an external inference server (TEI, vLLM, Triton):
  Replace _encode_sync with an httpx call and remove the ThreadPoolExecutor.
  The async interface (encode_documents / encode_query) stays identical.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import structlog
from sentence_transformers import SentenceTransformer

logger = structlog.get_logger()


class EmbeddingService:
    def __init__(
        self,
        model_name: str,
        local_models_dir: Path,
        device: str,
        query_prefix: str,
    ) -> None:
        self._prefix = query_prefix
        # max_workers=2: embedding is single-threaded inside torch; two slots
        # let one encode while the other waits for async I/O to resume.
        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="embedder")

        log = logger.bind(model=model_name, device=device)
        log.info("embedding_model_loading")

        local_path = local_models_dir / model_name.replace("/", "--")
        if local_path.is_dir():
            self._model = SentenceTransformer(str(local_path), device=device)
        else:
            os.makedirs(local_models_dir, exist_ok=True)
            self._model = SentenceTransformer(model_name, device=device)
            self._model.save(str(local_path))

        log.info("embedding_model_ready", dimension=self.dimension)

    @property
    def dimension(self) -> int:
        return self._model.get_sentence_embedding_dimension()  # type: ignore[return-value]

    def _encode_sync(self, texts: list[str], prefix: str) -> np.ndarray:
        return self._model.encode(
            [prefix + t for t in texts],
            normalize_embeddings=True,
            show_progress_bar=False,
        )

    async def encode_documents(self, texts: list[str]) -> np.ndarray:
        """Encode raw document texts — no instruction prefix."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode_sync, texts, "")

    async def encode_query(self, query: str) -> np.ndarray:
        """Encode a single search query with the BGE instruction prefix."""
        loop = asyncio.get_running_loop()
        return await loop.run_in_executor(self._executor, self._encode_sync, [query], self._prefix)

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
