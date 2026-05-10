from datetime import datetime, timezone
from enum import Enum
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class Role(str, Enum):
  USER = "user"
  ADMIN = "admin"


class ChatMessage(BaseModel):
  """A single message within a chat session."""
  messages: list[str]
  role: Role
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))


class Chat(BaseModel):
  __table__ = "chats"

  id: UUID = Field(default_factory=uuid4, alias="_id")
  chats: list[ChatMessage] = Field(default_factory=list)
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

  class Config:
    populate_by_name = True
