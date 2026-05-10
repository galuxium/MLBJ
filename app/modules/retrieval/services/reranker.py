"""
RerankerService — wraps CrossEncoder for async use.

To replace with an external reranking API (Cohere, Jina, TEI):
  Swap _predict_sync for an httpx call and remove the ThreadPoolExecutor.
  The async interface (rerank) stays identical.
"""
from __future__ import annotations

import asyncio
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import structlog
import torch
from sentence_transformers import CrossEncoder

logger = structlog.get_logger()


class RerankerService:
    def __init__(self, model_name: str, device: str, local_models_dir: Path) -> None:
        log = logger.bind(model=model_name, device=device)
        log.info("reranker_loading")

        local_path = local_models_dir / model_name.replace("/", "--")
        if local_path.is_dir():
            # local_files_only=True skips the ~10 HuggingFace HEAD requests
            # that otherwise verify cache freshness on every startup.
            self._model = CrossEncoder(str(local_path), device=device, local_files_only=True)
        else:
            os.makedirs(local_models_dir, exist_ok=True)
            self._model = CrossEncoder(model_name, device=device)
            self._model.save_pretrained(str(local_path))

        self._executor = ThreadPoolExecutor(max_workers=2, thread_name_prefix="reranker")
        log.info("reranker_ready")

    def _predict_sync(
        self, pairs: list[tuple[str, str]]
    ) -> tuple[list[float], list[float]]:
        logits = [float(x) for x in self._model.predict(pairs, batch_size=32)]
        # sigmoid is for display only; ranking uses raw logits to preserve score gaps
        probs = torch.sigmoid(torch.tensor(logits)).tolist()
        return logits, probs

    async def rerank(
        self,
        query: str,
        candidate_ids: list[str],
        candidate_texts: list[str],
        top_k: int,
    ) -> list[tuple[str, float, float]]:
        """
        Returns (case_no, logit, sigmoid_prob) sorted by logit descending.
        Always called with the original unexpanded query for precision.
        """
        if not candidate_ids:
            return []
        pairs = [(query, text) for text in candidate_texts]
        loop = asyncio.get_running_loop()
        logits, probs = await loop.run_in_executor(self._executor, self._predict_sync, pairs)
        ranked = sorted(zip(candidate_ids, logits, probs), key=lambda x: x[1], reverse=True)
        return list(ranked[:top_k])

    def shutdown(self) -> None:
        self._executor.shutdown(wait=False)
