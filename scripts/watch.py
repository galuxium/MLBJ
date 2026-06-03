import asyncio
import sys
from dotenv import load_dotenv

# Load environment variables first
load_dotenv()

from app.core.config import get_settings
from app.config.mongoClient import get_collection, close_mongo_client
from qdrant_client import AsyncQdrantClient
from qdrant_client.http.exceptions import UnexpectedResponse

async def main():
    settings = get_settings()
    
    # Initialize Qdrant Client
    qdrant = AsyncQdrantClient(
        url=settings.qdrant_url,
        api_key=settings.qdrant_api_key,
        timeout=10,
    )
    
    # MongoDB connection
    judgments_collection = await get_collection(settings.qdrant_collection)
    
    print(f"Watching synchronization progress for '{settings.qdrant_collection}'...")
    print("Press Ctrl+C to stop.\n")
    
    try:
        while True:
            # Get Mongo count
            try:
                mongo_count = await judgments_collection.count_documents({})
            except Exception as e:
                mongo_count = f"Error: {e}"
                
            # Get Qdrant count
            try:
                collection_info = await qdrant.get_collection(settings.qdrant_collection)
                qdrant_count = collection_info.points_count or 0
            except UnexpectedResponse as exc:
                if exc.status_code == 404:
                    qdrant_count = 0  # Not created yet
                else:
                    qdrant_count = f"Error: {exc.status_code}"
            except Exception as e:
                qdrant_count = f"Error: {e}"
            
            # Format and display the output over the same line
            output = f"\r[Status] MongoDB Cases: {mongo_count} | Qdrant Vectors: {qdrant_count}  "
            sys.stdout.write(output)
            sys.stdout.flush()
            
            await asyncio.sleep(2)
            
    except asyncio.CancelledError:
        pass
    finally:
        print("\nStopping watch...")
        await close_mongo_client()

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        pass
