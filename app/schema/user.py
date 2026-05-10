from datetime import datetime, timezone
from pydantic import BaseModel, EmailStr, Field


class User(BaseModel):
    __table__ = "users"

    id: str | None = Field(default=None, alias="_id")
    username: str = Field(...)
    email: EmailStr = Field(...)
    password: str = Field(...)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
