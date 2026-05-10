from app.schema.user import User
from app.schema.judgements import judgements
import asyncio
from dotenv import load_dotenv

load_dotenv()


async def setup():
    from app.config.mongoClient import get_database

    db = await get_database()

    existing_collections = await db.list_collection_names()

    if User.__table__ not in existing_collections:
        await db.create_collection(User.__table__)
        print(f"Created collection: {User.__table__}")
    else:
        print(f"Collection already exists: {User.__table__}")

    if judgements.__table__ not in existing_collections:
        await db.create_collection(judgements.__table__)
        print(f"Created collection: {judgements.__table__}")
    else:
        print(f"Collection already exists: {judgements.__table__}")


if __name__ == "__main__":
    asyncio.run(setup())
