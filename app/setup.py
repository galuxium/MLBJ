import asyncio
import os
import structlog
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from app.core.config import get_settings
from app.core.logging import configure_logging
from app.config.mongoClient import get_collection, close_mongo_client
from app.modules.retrieval.services.embedder import EmbeddingService
from app.modules.retrieval.services.sparse import SparseSearchService
from app.modules.retrieval.services.registry import DocumentRegistry
from app.modules.retrieval.services.indexing import IndexingService
from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams
from qdrant_client.http.exceptions import UnexpectedResponse

logger = structlog.get_logger()

async def _ensure_qdrant_collection(qdrant: AsyncQdrantClient, settings) -> None:
    """Create collection if it doesn't exist."""
    try:
        await qdrant.get_collection(settings.qdrant_collection)
    except UnexpectedResponse as exc:
        if exc.status_code == 404:
            logger.info("creating_qdrant_collection", collection=settings.qdrant_collection)
            await qdrant.create_collection(
                collection_name=settings.qdrant_collection,
                vectors_config=VectorParams(
                    size=1024,  # BGE-large dimensionality
                    distance=Distance.COSINE,
                ),
            )
        else:
            raise

async def main():
    logger.info("starting_setup_script")
    
    settings = get_settings()
    configure_logging(settings.log_level)
    
    if os.getenv("HF_HUB_OFFLINE") == "1":
        os.environ.setdefault("TRANSFORMERS_OFFLINE", "1")
        
    # Initialize Qdrant Client
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=120,
    )
    
    await _ensure_qdrant_collection(qdrant, settings)
    
    # Initialize components
    embedder = EmbeddingService(
        model_name=settings.embedding_model,
        local_models_dir=settings.local_models_dir,
        device=settings.device,
        query_prefix=settings.bge_query_prefix,
    )
    
    sparse = SparseSearchService()
    registry = DocumentRegistry()
    
    # MongoDB connection
    judgments_collection = await get_collection("judgements")
    
    # Initialize the actual Indexing Service which contains load_from_mongo
    indexing_svc = IndexingService(
        embedder=embedder,
        sparse=sparse,
        registry=registry,
        qdrant=qdrant,
        mongo_collection=judgments_collection,
        settings=settings,
    )
    
    logger.info("running_load_from_mongo_process")
    await indexing_svc.load_from_mongo()
    logger.info("setup_completed")
    
    # Cleanup properly
    await close_mongo_client()

if __name__ == "__main__":
    asyncio.run(main())