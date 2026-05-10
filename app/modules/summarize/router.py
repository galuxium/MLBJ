import logging
import os
from datetime import datetime, timezone

from fastapi import APIRouter, Depends, HTTPException, UploadFile, status
from fastapi.responses import JSONResponse
from google import genai
from google.genai import types
from pydantic import BaseModel, field_validator
from pypdf import PdfReader

from app.modules.auth.router import get_current_user
from .prompts import retrievePrompts

logger = logging.getLogger(__name__)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
  raise RuntimeError("GEMINI_API_KEY environment variable is not set.")

GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.0-flash")
MAX_FILE_SIZE_BYTES = 10 * 1024 * 1024  # 10 MB
ALLOWED_EXTENSIONS = {".pdf"}

router = APIRouter(prefix="/summarize", tags=["Summarization"])
_genai_client = genai.Client(api_key=GEMINI_API_KEY)
_prompts = retrievePrompts()


class TextRequest(BaseModel):
  query: str

  @field_validator("query")
  @classmethod
  def query_must_not_be_empty(cls, v: str) -> str:
    if not v.strip():
      raise ValueError("query must not be empty.")
    return v


@router.get("/health")
def health_check():
  return JSONResponse(content={"status": "ok"})


def _extract_with_gemini(system_prompt: str, user_prompt: str) -> str:
  response = _genai_client.models.generate_content(
    model=GEMINI_MODEL,
    config=types.GenerateContentConfig(
      temperature=0,
      thinking_config=types.ThinkingConfig(thinking_level="medium"),
    ),
    contents=[
      types.Content(role="system", parts=[types.Part(text=system_prompt)]),
      types.Content(role="user", parts=[types.Part(text=user_prompt)]),
    ],
  )
  return response.text


def _format_response(raw: str) -> dict:
  """Parse the delimited plain-text Gemini response into a structured dict."""
  final: dict = {}
  for section in raw.split("###"):
    section = section.strip()
    if not section:
      continue
    lines = section.splitlines()
    header = lines[0].replace(":", "").strip()
    body = "\n".join(lines[1:]).strip()
    points = [
      p.strip().lstrip("- ").strip()
      for p in body.split("***")
      if p.strip()
    ]
    final[header] = points
  return final


@router.post("/text")
async def summarize_text(
  data: TextRequest,
  _: dict = Depends(get_current_user),
):
  """Extract structured case information from a text description."""
  try:
    raw = _extract_with_gemini(
      system_prompt=_prompts.system_prompt(),
      user_prompt=_prompts.user_prompt(text=data.query),
    )
    return JSONResponse(
      content={
        "result": _format_response(raw),
        "timestamp": datetime.now(timezone.utc).isoformat(),
      },
      status_code=200,
    )
  except Exception:
    logger.exception("Gemini extraction failed for /summarize/text.")
    raise HTTPException(
      status_code=status.HTTP_502_BAD_GATEWAY,
      detail="Upstream AI service error.",
    )


@router.post("/file")
async def extract_from_file(
  file: UploadFile,
  _: dict = Depends(get_current_user),
):
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
    reader = PdfReader(io.BytesIO(content))
    text = "\n".join(page.extract_text() or "" for page in reader.pages)
    return JSONResponse(
      content={
        "data": text,
        "file": file.filename,
        "timestamp": datetime.now(timezone.utc).isoformat(),
      },
      status_code=200,
    )
  except Exception:
    logger.exception("PDF extraction failed for file '%s'.", file.filename)
    raise HTTPException(status_code=422, detail="Could not parse PDF.")
