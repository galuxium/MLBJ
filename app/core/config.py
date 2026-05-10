from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    # ── MongoDB ───────────────────────────────────────────────────────────
    mongodb_uri: str
    database_name: str

    # ── Qdrant (dense vector store) ───────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str | None = None
    qdrant_collection: str = "judgements"

    # ── Inference models ──────────────────────────────────────────────────
    embedding_model: str = "BAAI/bge-large-en-v1.5"
    cross_encoder_model: str = "BAAI/bge-reranker-v2-m3"
    # BGE instruction prefix — applied to queries only, not to documents
    bge_query_prefix: str = "Represent this sentence for searching relevant passages: "
    local_models_dir: Path = Path("llm-models")
    device: str = "cpu"

    # ── Retrieval tuning ──────────────────────────────────────────────────
    dense_weight: float = 0.6
    shortlist_multiplier: int = 4
    use_query_expansion: bool = True
    max_syns_per_token: int = 2
    max_hypos_per_token: int = 2

    # ── Auth ──────────────────────────────────────────────────────────────
    jwt_secret_key: str
    jwt_algorithm: str = "HS256"
    jwt_expire_minutes: int = 60

    # ── Gemini ────────────────────────────────────────────────────────────
    gemini_api_key: str
    gemini_model: str = "gemini-2.0-flash"

    # ── Application ───────────────────────────────────────────────────────
    env: str = "development"
    cors_origins: str = "http://localhost:3000"
    log_level: str = "INFO"

    model_config = SettingsConfigDict(
        env_file=".env",
        extra="ignore",
        case_sensitive=False,
    )


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    return Settings()
