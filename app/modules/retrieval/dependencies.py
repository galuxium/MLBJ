"""
FastAPI dependency providers for the retrieval module.

All services are stored on `app.state` (set in main.py lifespan) and
extracted here via Request injection. This pattern makes routes testable:
swap `app.state.*` in pytest fixtures and the Depends() calls resolve
to the test doubles automatically.
"""
from __future__ import annotations

from fastapi import Request

from .services.indexing import IndexingService
from .services.registry import DocumentRegistry
from .services.search import SearchService


def get_search_service(request: Request) -> SearchService:
    return request.app.state.search_service


def get_indexing_service(request: Request) -> IndexingService:
    return request.app.state.indexing_service


def get_registry(request: Request) -> DocumentRegistry:
    return request.app.state.document_registry
