from datetime import datetime, timezone

from pydantic import BaseModel, Field


class judgements(BaseModel):
  __table__ = "judgements"

  id: str | None = Field(default=None, alias="_id")
  case_no: str = Field(...)
  title: str = Field(...)
  jurisdiction: str = Field(...)
  date: datetime = Field(...)
  issues: list[str] = Field(...)
  facts: list[str] = Field(...)
  court_reasoning: list[str] = Field(...)
  precedent_analysis: list[str] = Field(...)
  argument_by_petitioner: list[str] = Field(...)
  conclusion: list[str] = Field(...)
  ipc_sections: str | None = Field(default=None)
  statute_analysis: list[str] = Field(...)
  argument_by_respondent: list[str] = Field(...)
  created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

  class Config:
    populate_by_name = True
