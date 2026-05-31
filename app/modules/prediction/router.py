from fastapi import APIRouter
from fastapi.responses import JSONResponse
from qdrant_client.models import BaseModel
from .services import predict_judgment

router = APIRouter(prefix="/predict", tags=["Prediction"])

class RequestBody(BaseModel):
    text: str

class ResponseBody(BaseModel):
    prediction: int
    label: str
    confidence: float
    logits: list[float]
    probabilities: list[float]
    n_chunks_used: int

@router.get("/health")
def health_check():
    return JSONResponse(content={"status": "ok"}, status_code=200)


def clean_text(text: str) -> str:
    """Basic cleaning to remove excessive whitespace."""
    UNWANTED_CHARACTERS = ["\n", "\t", "\r"]
    for char in UNWANTED_CHARACTERS:
        text = text.replace(char, " ")
    return ' '.join(text.split())


@router.post("/text", response_model=ResponseBody)
async def predict_text(request_body: RequestBody):
    request_text = request_body.text
    request_text = clean_text(request_text)
    result = predict_judgment(request_text)
    return JSONResponse(content=result, status_code=200)


@router.post("/file", response_model=ResponseBody)
async def predict_file(file: bytes):
    text = file.decode('utf-8')
    text = clean_text(text)
    result = predict_judgment(text)
    return JSONResponse(content=result, status_code=200)