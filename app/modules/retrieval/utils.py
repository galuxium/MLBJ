"""
Pure, stateless helpers shared across retrieval services.
No imports from FastAPI, no side effects.
"""
from __future__ import annotations

import re
import uuid
from collections import Counter
from typing import Any

import nltk

for _pkg, _path in (
    ("stopwords", "corpora/stopwords"),
    ("wordnet",   "corpora/wordnet"),
    ("omw-1.4",   "corpora/omw-1.4"),
):
    try:
        nltk.data.find(_path)
    except LookupError:
        nltk.download(_pkg, quiet=True)

from nltk.corpus import stopwords as _nltk_sw, wordnet

STOPWORDS: set[str] = set(_nltk_sw.words("english"))

SEMANTIC_FIELDS = [
    "issue", "facts", "court_reasoning", "precedent_analysis",
    "argument_by_petitioner", "argument_by_respondent",
    "statute_analysis", "conclusion", "ipc_sections",
]

_WORD_RE = re.compile(r"[a-z]+")


# ── Document text ─────────────────────────────────────────────────────────────

def combine_text(data: dict[str, Any]) -> str:
    """Concatenate all searchable text fields into one string for embedding/BM25."""
    fields = [
        data.get("title", ""), data.get("issue"), data.get("facts"),
        data.get("court_reasoning"), data.get("precedent_analysis"),
        data.get("argument_by_petitioner"), data.get("conclusion"),
        data.get("statute_analysis"), data.get("argument_by_respondent"),
    ]
    parts: list[str] = []
    for f in fields:
        if f is None:
            continue
        if isinstance(f, list):
            parts.extend(str(i) for i in f if i)
        else:
            parts.append(str(f))
    return " ".join(parts)


# ── Stable IDs ────────────────────────────────────────────────────────────────

def stable_point_id(case_no: str) -> str:
    """Deterministic UUID v5 from case_no — used as the Qdrant point ID."""
    return str(uuid.uuid5(uuid.NAMESPACE_DNS, case_no))


# ── Query expansion ───────────────────────────────────────────────────────────

def expand_query(
    query: str,
    max_syns: int = 2,
    max_hypos: int = 2,
) -> str:
    """
    Append WordNet synonyms + hyponyms so that 'weapon' surfaces 'knife', etc.
    Applied to the retrieval (dense + BM25) query only — the cross-encoder
    always receives the original unexpanded query for precise reranking.
    """
    seen: set[str] = set(_WORD_RE.findall(query.lower()))
    extra: list[str] = []

    for token in _WORD_RE.findall(query.lower()):
        if token in STOPWORDS or len(token) < 3:
            continue
        syns_added = hypos_added = 0
        for synset in wordnet.synsets(token):
            for lemma in synset.lemmas():
                w = lemma.name().replace("_", " ").lower()
                if w == token or w in seen or " " in w:
                    continue
                extra.append(w)
                seen.add(w)
                syns_added += 1
                if syns_added >= max_syns:
                    break
            for hypo in synset.hyponyms():
                for lemma in hypo.lemmas():
                    w = lemma.name().replace("_", " ").lower()
                    if w == token or w in seen or " " in w:
                        continue
                    extra.append(w)
                    seen.add(w)
                    hypos_added += 1
                    if hypos_added >= max_hypos:
                        break
                if hypos_added >= max_hypos:
                    break
            if syns_added >= max_syns and hypos_added >= max_hypos:
                break

    return query + " " + " ".join(extra) if extra else query


# ── Fusion ────────────────────────────────────────────────────────────────────

def minmax_normalize(scores: dict[str, float]) -> dict[str, float]:
    if not scores:
        return {}
    vals = list(scores.values())
    lo, hi = min(vals), max(vals)
    if hi - lo < 1e-9:
        return {k: 0.0 for k in scores}
    return {k: (v - lo) / (hi - lo) for k, v in scores.items()}


def weighted_fusion(
    dense: dict[str, float],
    sparse: dict[str, float],
    dense_weight: float,
) -> dict[str, float]:
    """Min-max-normalised weighted combination. Preserves score magnitudes unlike RRF."""
    dn = minmax_normalize(dense)
    sn = minmax_normalize(sparse)
    return {
        doc_id: dense_weight * dn.get(doc_id, 0.0) + (1.0 - dense_weight) * sn.get(doc_id, 0.0)
        for doc_id in set(dn) | set(sn)
    }


def reciprocal_rank_fusion(
    dense: dict[str, float],
    sparse: dict[str, float],
    k: int = 60,
) -> dict[str, float]:
    fused: dict[str, float] = {}
    for rank, (doc_id, _) in enumerate(
        sorted(dense.items(), key=lambda x: x[1], reverse=True), start=1
    ):
        fused[doc_id] = fused.get(doc_id, 0) + 1.0 / (k + rank)
    for rank, (doc_id, _) in enumerate(
        sorted(sparse.items(), key=lambda x: x[1], reverse=True), start=1
    ):
        fused[doc_id] = fused.get(doc_id, 0) + 1.0 / (k + rank)
    return fused


# ── Semantic representation ───────────────────────────────────────────────────

def extract_semantic_terms(text: str) -> list[str]:
    terms = re.findall(r"[a-zA-Z]{3,}", text.lower())
    return [t for t in terms if t not in STOPWORDS]


def normalize_field(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, list):
        return " ".join(str(i) for i in value if i)
    return str(value)


def build_semantic_representation(
    judgments: list[dict[str, Any]],
    max_terms: int = 10,
) -> dict[str, list[str]]:
    if not judgments:
        return {}
    min_freq = 2 if len(judgments) > 1 else 1
    result: dict[str, list[str]] = {}
    for field in SEMANTIC_FIELDS:
        freq: Counter = Counter()
        for j in judgments:
            text = normalize_field(j.get(field))
            if text:
                for term in set(extract_semantic_terms(text)):
                    freq[term] += 1
        common = [t for t, c in freq.most_common() if c >= min_freq][:max_terms]
        if common:
            result[field] = common
    return result
