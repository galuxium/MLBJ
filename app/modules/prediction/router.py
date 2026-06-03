from fastapi import APIRouter, HTTPException, UploadFile
from fastapi.responses import JSONResponse
from qdrant_client.models import BaseModel
from .services import predict_judgment
import os

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

MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf"}

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
async def predict_file(file: UploadFile):
    """Extract raw text from an uploaded PDF."""
    if file.filename is None:
        raise HTTPException(status_code=400, detail="No filename provided.")

    ext = os.path.splitext(file.filename)[-1].lower()
    if ext not in ALLOWED_EXTENSIONS:
        raise HTTPException(
        status_code=400,
        detail=f"Unsupported file type '{ext}'. Allowed: {sorted(ALLOWED_EXTENSIONS)}",
        )

    # Guard against oversized uploads before reading into memory
    content = await file.read()
    if len(content) > MAX_FILE_SIZE_BYTES:
        raise HTTPException(
        status_code=413,
        detail=f"File exceeds the {MAX_FILE_SIZE_BYTES // (1024 * 1024)} MB limit.",
        )

    try:
        import io
        from pypdf import PdfReader
        reader = PdfReader(io.BytesIO(content))
        text = "\n".join(page.extract_text() or "" for page in reader.pages)
        text = clean_text(text)
        result = predict_judgment(text)
        return JSONResponse(content=result, status_code=200)
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Error processing PDF: {str(e)}")