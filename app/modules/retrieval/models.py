from __future__ import annotations

from enum import Enum
from typing import List, Optional

from pydantic import BaseModel, Field


# ── Domain ────────────────────────────────────────────────────────────────────

class Judgment(BaseModel):
    case_no: str
    title: str
    jurisdiction: Optional[str] = None
    date: Optional[str] = None
    issue: Optional[List[str]] = None
    facts: Optional[List[str]] = None
    court_reasoning: Optional[List[str]] = None
    precedent_analysis: Optional[List[str]] = None
    argument_by_petitioner: Optional[List[str]] = None
    conclusion: Optional[str] = None
    ipc_sections: Optional[List[str]] = None
    statute_analysis: Optional[List[str]] = None
    argument_by_respondent: Optional[List[str]] = None

    def get_combined_text(self) -> str:
        fields = [
            self.title, self.issue, self.facts, self.court_reasoning,
            self.precedent_analysis, self.argument_by_petitioner,
            self.conclusion, self.statute_analysis, self.argument_by_respondent,
        ]
        parts: list[str] = []
        for f in fields:
            if f is None:
                continue
            parts.append(str(f))
        return " ".join(parts)


class JudgmentBatch(BaseModel):
    judgments: list[Judgment]


# ── Request ───────────────────────────────────────────────────────────────────

class FusionStrategy(str, Enum):
    RRF      = "rrf"
    WEIGHTED = "weighted"


class SearchQuery(BaseModel):
    query: str
    top_k: int = Field(default=10, ge=1, le=100)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    fusion: FusionStrategy = FusionStrategy.WEIGHTED
    enable_rerank: bool = Field(default=True)
    expand_query: bool = Field(default=True)


class SearchDocQuery(BaseModel):
    query: Judgment
    top_k: int = Field(default=10, ge=1, le=100)
    dense_weight: float = Field(default=0.6, ge=0.0, le=1.0)
    enable_rerank: bool = Field(default=True)


# ── Response ──────────────────────────────────────────────────────────────────

class SearchResult(BaseModel):
    case_no: str
    title: str
    jurisdiction: Optional[str] = None
    date: Optional[str] = None
    issue: Optional[List[str]] = None
    facts: Optional[List[str]] = None
    court_reasoning: Optional[List[str]] = None
    precedent_analysis: Optional[List[str]] = None
    argument_by_petitioner: Optional[List[str]] = None
    conclusion: Optional[List[str]] = None
    ipc_sections: Optional[List[str]] = None
    statute_analysis: Optional[List[str]] = None
    argument_by_respondent: Optional[List[str]] = None
    dense_score: float
    sparse_score: float
    combined_score: float
    rerank_score: Optional[float] = None   # raw cross-encoder logit
    rerank_prob: Optional[float] = None    # sigmoid(logit) — display only


class SearchResponse(BaseModel):
    results: list[SearchResult]
    query: str
    total_results: int


class SearchDocResponse(BaseModel):
    results: list[SearchResult]
    query: Judgment
    total_results: int
    semantic_representation: dict[str, list[str]]


class IndexResponse(BaseModel):
    status: str
    message: str
