import logging
import os
from typing import Optional

from pymongo import AsyncMongoClient, server_api
from pymongo.asynchronous.collection import AsyncCollection
from pymongo.asynchronous.database import AsyncDatabase


logger = logging.getLogger(__name__)


def _get_required_env(key: str) -> str:
    value = os.getenv(key)
    if not value:
        logger.error("%s environment variable is not set.", key)
        raise ValueError(f"{key} environment variable is not set.")
    return value


MONGODB_URI: Optional[str] = None
DATABASE_NAME: Optional[str] = None


_mongo_client: Optional[AsyncMongoClient] = None


def _ensure_settings_loaded() -> None:
    global MONGODB_URI, DATABASE_NAME

    if MONGODB_URI is None:
        MONGODB_URI = _get_required_env("MONGODB_URI")
    if DATABASE_NAME is None:
        DATABASE_NAME = _get_required_env("DATABASE_NAME")


async def get_mongo_client() -> AsyncMongoClient:
    global _mongo_client
    _ensure_settings_loaded()
    assert MONGODB_URI is not None

    if _mongo_client is None:
        _mongo_client = AsyncMongoClient(
            MONGODB_URI,
            server_api=server_api.ServerApi(
                version="1", strict=True, deprecation_errors=True
            ),
        )
        try:
            await _mongo_client.admin.command("ping")
        except Exception:
            _mongo_client.close()
            _mongo_client = None
            logger.exception("Error connecting to MongoDB.")
            raise

    return _mongo_client


async def get_database() -> AsyncDatabase:
    _ensure_settings_loaded()
    assert DATABASE_NAME is not None
    client = await get_mongo_client()
    return client.get_database(DATABASE_NAME)


async def get_collection(collection_name: str) -> AsyncCollection:
    db = await get_database()
    return db.get_collection(collection_name)


async def close_mongo_client() -> None:
    global _mongo_client

    if _mongo_client is not None:
        _mongo_client.close()
        _mongo_client = None
