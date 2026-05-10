from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse

router = APIRouter(prefix="/chatbot", tags=["Chatbot"])


@router.get("/health")
async def health_check():
    return JSONResponse(content={"status": "healthy"}, status_code=200)


@router.post("/ask")
async def ask_question():
    return JSONResponse(content={"message": "Question received"}, status_code=200)
