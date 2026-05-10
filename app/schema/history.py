from datetime import datetime, timezone
from uuid import UUID, uuid4

from pydantic import BaseModel, Field


class History(BaseModel):
  __table__ = "history"

  id: UUID = Field(default_factory=uuid4, alias="_id")
  chat_ids: list[UUID] = Field(...)
  user_id: UUID = Field(...)
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

  class Config:
    populate_by_name = True
