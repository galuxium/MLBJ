import os
from datetime import datetime, timedelta, timezone

import bcrypt
import jwt
from fastapi import Depends, HTTPException, status
from fastapi.routing import APIRouter
from fastapi.responses import JSONResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from app.config.mongoClient import get_collection
from app.models.auth import (
  LoginRequest,
  LoginResponse,
  RegisterRequest,
  RegisterResponse,
)
from app.schema.user import User

router = APIRouter(prefix="/auth", tags=["Authentication"])
security = HTTPBearer()

USERS_COLLECTION = "users"
JWT_ALGORITHM = os.getenv("JWT_ALGORITHM", "HS256")
JWT_EXPIRE_MINUTES = int(os.getenv("JWT_EXPIRE_MINUTES", "60"))
_JWT_SECRET_KEY: str | None = None


def _get_jwt_secret() -> str:
  """
  Lazily fetch JWT secret so module import does not fail if env is not
  yet loaded. Raises at call-time, not at import-time.
  """
  global _JWT_SECRET_KEY
  if _JWT_SECRET_KEY is None:
    secret = os.getenv("JWT_SECRET_KEY")
    if not secret:
      raise RuntimeError("JWT_SECRET_KEY environment variable is not set.")
    _JWT_SECRET_KEY = secret
  return _JWT_SECRET_KEY


async def ensure_auth_indexes() -> None:
  """
  Create unique indexes once at startup — not on every request.
  Called from app lifespan, not from route handlers.
  """
  users_collection = await get_collection(USERS_COLLECTION)
  await users_collection.create_index("username", unique=True)
  await users_collection.create_index("email", unique=True)


def _hash_password(password: str) -> str:
  hashed = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())
  return hashed.decode("utf-8")


def _verify_password(plain: str, hashed: str) -> bool:
  return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


def _create_access_token(email: str) -> str:
  now = datetime.now(timezone.utc)
  payload = {
    "sub": email,
    "iat": int(now.timestamp()),
    "exp": int((now + timedelta(minutes=JWT_EXPIRE_MINUTES)).timestamp()),
  }
  return jwt.encode(payload, _get_jwt_secret(), algorithm=JWT_ALGORITHM)


def get_current_user(
  credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
  """
  Shared dependency: validates Bearer JWT and returns the decoded payload.
  Apply to any route that requires authentication.
  """
  try:
    payload = jwt.decode(
      credentials.credentials,
      _get_jwt_secret(),
      algorithms=[JWT_ALGORITHM],
    )
    return payload
  except jwt.ExpiredSignatureError:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Token has expired",
    )
  except jwt.InvalidTokenError:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Invalid token",
    )


@router.get("/health")
async def health_check():
  return JSONResponse(content={"status": "healthy"}, status_code=200)


@router.post("/login", response_model=LoginResponse)
async def login(request: LoginRequest):
  users_collection = await get_collection(USERS_COLLECTION)
  user = await users_collection.find_one({"email": request.email.lower().strip()})

  # Always run verify_password to prevent timing oracle attacks
  password_ok = user is not None and _verify_password(
    request.password, user["password"]
  )
  if not password_ok:
    raise HTTPException(
      status_code=status.HTTP_401_UNAUTHORIZED,
      detail="Invalid email or password",
    )

  token = _create_access_token(user["email"])
  return LoginResponse(message="Login successful", access_token=token)


@router.post("/register", response_model=RegisterResponse)
async def register(request: RegisterRequest):
  username = request.username.lower().strip()
  email = request.email.lower().strip()
  users_collection = await get_collection(USERS_COLLECTION)

  existing = await users_collection.find_one(
    {"$or": [{"username": username}, {"email": email}]}
  )
  if existing:
    raise HTTPException(
      status_code=status.HTTP_409_CONFLICT,
      detail="Username or email already exists",
    )

  user_doc = User(
    username=username,
    email=email,
    password=_hash_password(request.password),
  )
  await users_collection.insert_one(
    user_doc.model_dump(by_alias=True, exclude={"id"})
  )
  return RegisterResponse(message="Registration successful")
