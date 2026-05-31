# Changelog

All changes to the codebase are documented here.

---

## [2026-05-13] Code Review & Security Hardening

### Summary
Comprehensive principal-engineer code review completed. Applied fixes across security, correctness, architecture, and code quality. All 8 core files refactored.

**Security fixes:** Credentials rotation required (live in `.env`). JWT auth enforcement added. File upload guards added. Error detail leakage eliminated.

**Correctness fixes:** Silent route collision fixed. Broken schemas fixed. Typo propagated throughout codebase fixed.

**Architecture:** Thread-safety locks added. Deprecated async APIs updated. Mutable state serialized. Search side-effects removed. Magic-number fusion replaced with enum.

---

### Files Modified

#### `app/modules/auth/router.py`
- **[NEW]** `ensure_auth_indexes()` — moved DB index creation from per-request to startup
- **[NEW]** `_get_jwt_secret()` — lazy loading with hard failure if env var missing (no silent fallback)
- **[NEW]** `get_current_user(credentials: HTTPAuthorizationCredentials)` — shared JWT dependency for protected routes
- **[CHANGED]** Password verification now always runs (timing-oracle attack mitigation)
- **[REMOVED]** `_get_users_collection()` — indexes no longer created per-request

#### `app/main.py`
- **[NEW]** `CORSMiddleware` with `CORS_ORIGINS` env var
- **[CHANGED]** Lifespan now calls `ensure_auth_indexes()` at startup
- **[NEW]** Structured logging via `basicConfig`
- **[NEW]** Docs hidden in production if `ENV != "production"`
- **[CHANGED]** Message: `"Welcome to the FastAPI app!"` → `"Judicio AI is running."`

#### `app/modules/retrieval/router.py`
- **[NEW]** `asyncio.Lock` (`_store_lock`) guards all mutations to `document_store`, `bm25_index`, `doc_ids_list`
- **[CHANGED]** `asyncio.get_event_loop()` → `asyncio.get_running_loop()` (Python 3.10+ compliant)
- **[REMOVED]** All `print()` calls; replaced with `logger.*`
- **[REMOVED]** Dead imports (`FastAPI`, `CORSMiddleware`)
- **[NEW]** `FusionStrategy` enum (`rrf` / `weighted`) — replaces magic-number fusion selector
- **[NEW]** `top_k` bounded via `Field(ge=1, le=100)` on `SearchQuery` and `SearchDocQuery`
- **[CHANGED]** `/search/hybrid/docs` now returns `404` instead of silently indexing — search never mutates state
- **[NEW]** `_judgment_to_search_result()` helper — eliminates 14-field copy-paste across 4 endpoints
- **[CHANGED]** Error responses no longer leak `str(e)` to clients
- **[NEW]** `_reciprocal_rank_fusion()`, `_weighted_fusion()`, `_dense_search()`, `_sparse_search()` — pure functions
- **[NEW]** `_rebuild_bm25()` — returns new tuple, caller assigns inside lock

#### `app/modules/summarize/router.py`
- **[FIXED]** Route collision: `/file` handler renamed from `retrieve_cases` → `extract_from_file`
- **[CHANGED]** Gemini model: `"gemini-3-flash-preview"` → `"gemini-2.0-flash"` (via `GEMINI_MODEL` env var)
- **[CHANGED]** `TextRequest.query` now `str`-only (no bytes)
- **[NEW]** `@field_validator` on `query` — ensures non-empty
- **[NEW]** 10 MB file size guard on upload
- **[CHANGED]** `file.filename` now null-checked before `splitext()`
- **[NEW]** `Depends(get_current_user)` applied to both `/text` and `/file` routes
- **[REMOVED]** Dead imports (`Depends`, `HTTPBearer` — were imported but unused)

#### `app/modules/summarize/prompts.py`
- **[FIXED]** Typo: `"Precident_Analysis"` → `"Precedent_Analysis"` in `SummaryContents`
- **[CHANGED]** `user_prompt(text: str | object | bytes)` → `user_prompt(text: str)` — str-only to match router
- **[REFACTORED]** Prompts are now formatted cleanly, line-by-line

#### `app/schema/judgements.py`
- **[FIXED]** Typo: `precident_analysis` → `precedent_analysis`
- **[CHANGED]** `ipc_sections: str` → `ipc_sections: str | None` (matches retrieval model)
- **[NEW]** `Config.populate_by_name = True` (Pydantic v2)

#### `app/schema/chats.py`
- **[FIXED]** `chatSchema` was a plain class → now `ChatMessage(BaseModel)`
- **[CHANGED]** `Role` now `Role(str, Enum)` for JSON serialisation
- **[CHANGED]** `_id` from bare annotation → proper `Field(default_factory=uuid4)`
- **[NEW]** `Config.populate_by_name = True`

#### `app/schema/history.py`
- **[FIXED]** `default_factory=UUID` (broken, raises `TypeError`) → `default_factory=uuid4`
- **[CHANGED]** `created_at: str` → `created_at: datetime`
- **[NEW]** `Config.populate_by_name = True`

---

### Security Considerations (Not Yet Done)

⚠️ **Action Required:**
1. **Rotate all credentials** — `.env` file contains live MongoDB password, JWT secret, Gemini API key, and HF token. These must be invalidated and replaced immediately. The old values are permanently in git history.
2. Move secrets to a secrets manager (AWS Secrets Manager, HashiCorp Vault, etc.).
3. Never commit `.env` to version control, even by mistake.

---

### Testing Recommendations

After these changes, verify:
- [ ] Auth flow: register, login, token generation/validation
- [ ] Protected routes: attempt access without token (should 401)
- [ ] Search endpoints: hybrid, dense-only, sparse-only with various `top_k` values
- [ ] File upload: valid PDF, oversized file (>10MB), invalid extension
- [ ] DB indexes: confirm unique constraint on email/username
- [ ] Concurrent writes: index multiple batches in parallel (lock prevents data corruption)

---

### Notes for Future Changes

When modifying the codebase going forward:
1. Keep route handlers thin — delegate to service/helper functions
2. All mutations to shared state must be guarded by `_store_lock`
3. Use `asyncio.get_running_loop()` in async contexts, not `asyncio.get_event_loop()`
4. Log structured data with `logger.*`, never `print()`
5. Never leak internal error details to HTTP clients — log server-side, return generic message
6. Add bounds validation to numeric inputs (`Field(ge=1, le=100)`)
7. Search operations must be read-only — no side-effecting indexing during search
8. Pydantic v2: use `populate_by_name = True` in `Config` for alias support

---

## [2026-05-13] Startup Performance & Bulk Embedding

### Summary
Cut cold-start time from ~45 s to ~3 s on warm restarts by persisting the registry and BM25 index to disk. Added a Colab GPU bulk-embed script so the one-time embedding of 5 022 docs takes ~3 min on T4 instead of hours on CPU. Fixed a Qdrant client timeout that crashed the startup hydration task, and migrated the dense search call off a now-removed `qdrant-client` method.

**Why this matters:** Every server restart was repeating the same deterministic 30–45 s of work (Mongo streaming + tokenization + BM25 IDF build) — wasteful, frustrating during development, and made deploys slow. Embeddings on CPU also blocked indexing for hours. The reranker silently made ~10 HTTP HEAD requests to HuggingFace on every startup, even though its weights were cached locally.

---

### Files Modified

#### `app/main.py`
- **[FIX]** Added missing `import asyncio` — `lifespan()` called `asyncio.create_task()` without importing the module, crashing startup.
- **[CHANGED]** `AsyncQdrantClient(..., timeout=120)` — default ~5 s timeout was too short; CPU embedding of a single batch can take 30–60 s, causing `ReadTimeout` mid-upsert.
- **[NEW]** Honors `HF_HUB_OFFLINE=1` env var → sets `TRANSFORMERS_OFFLINE=1`, so once models are cached locally, startup makes zero network calls.
- **[CHANGED]** Passes `local_models_dir` to `RerankerService` so it can persist the reranker on disk.

#### `app/modules/retrieval/services/reranker.py`
- **[NEW]** Mirrors the embedder's save/load pattern: on first run, downloads + saves to `llm-models/BAAI--bge-reranker-v2-m3/`; on subsequent runs, loads from disk with `local_files_only=True`.
- **[WHY]** Eliminated ~10 HuggingFace HEAD requests per startup. Reranker load time: ~10 s → ~0.5 s.

#### `app/modules/retrieval/services/search.py`
- **[FIX]** `_dense_search` now calls `qdrant.query_points()` and reads `response.points`. The old `AsyncQdrantClient.search()` method was removed in recent `qdrant-client` versions; every hybrid/dense request was returning 500.

#### `app/modules/retrieval/services/indexing.py`
- **[NEW]** Disk cache for `DocumentRegistry` + BM25 at `llm-models/index_cache.pkl`.
  - On startup: if Qdrant count matches MongoDB *and* the cache signature (sha256 of sorted case_nos) matches → load pickle (~2 s). Otherwise fall back to the existing slow path and write a fresh cache at the end.
  - On every successful `index_batch` and `delete` → cache is rewritten atomically (tmp file + `replace()`).
  - Defensive: pickle load failures, schema mismatches, and Qdrant-count mismatches all force a rebuild.
- **[CHANGED]** Startup batch size: 100 → 32. Smaller batches keep each embed+upsert cycle inside the Qdrant timeout window.
- **[NEW]** `ResponseHandlingException` is caught in both `_qdrant_upsert` and the `load_from_mongo` loop. Previously a transient Qdrant timeout would silently kill the background hydration task (`Task exception was never retrieved`) with no log.
- **[NEW]** Helper functions: `_fetch_mongo_case_nos()`, `_try_load_cache()`, `_save_cache()`, `_refresh_cache()`, module-level `_compute_signature()`.

#### `app/modules/retrieval/services/registry.py`
- **[NEW]** `case_nos()`, `dump_state()`, `load_state()` — expose the registry's internal dicts so they can be pickled and restored. Used exclusively by the IndexingService cache layer.

#### `app/modules/retrieval/services/sparse.py`
- **[NEW]** `dump_state()`, `load_state()` — expose the BM25 instance and its case_no order for the disk cache. `BM25Okapi` is pure-Python and pickles cleanly.

#### `scripts/colab_bulk_embed.py` *(new file)*
- One-shot Colab GPU script for bulk-embedding all docs from MongoDB into Qdrant.
- Mirrors `combine_text` and `stable_point_id` from `app/modules/retrieval/utils.py` exactly, so the points it writes are byte-for-byte compatible with what the backend expects.
- Idempotent (deterministic UUIDs), retry logic on upsert, 120 s timeout, progress bars, GPU verification.
- **WHY:** Embedding 5 022 docs on CPU takes hours and frequently times out. Free Colab T4 finishes in ~3 min.

---

### Cache Invariants

| Trigger | Behavior |
|---|---|
| Case added or removed in Mongo | Signature mismatch → cache rebuilt next startup |
| `POST /retrieval/judgments/index` | Cache rewritten after successful index |
| `DELETE /retrieval/judgments/{case_no}` | Cache rewritten after successful delete |
| Qdrant count ≠ Mongo count | Cache **not** trusted even if signature matches → full re-embed path |
| Pickle corrupt or schema drift | Logs warning, falls back to slow path, writes fresh cache |
| Doc text edited in Mongo *without* changing `case_no` | Signature unchanged → stale cache loaded. **Workaround:** delete `llm-models/index_cache.pkl` manually |

---

### Performance

| Phase | Before | After (cold) | After (warm) |
|---|---|---|---|
| Embedder load | 0.5 s | 0.5 s | 0.5 s |
| Reranker load | ~10 s | ~10 s | ~0.5 s |
| Mongo hydration | ~10 s | ~10 s | 0 s (cached) |
| BM25 build | ~30 s | ~30 s | 0 s (cached) |
| Pickle load | — | — | ~1–2 s |
| **Total to ready** | **~45 s** | **~45 s (1st run only)** | **~3 s** |

---

### Operator Notes
1. After deploying, the **first** restart still does the full work and writes `llm-models/index_cache.pkl`. **Every restart after that is fast.**
2. Once both `llm-models/BAAI--bge-large-en-v1.5/` and `llm-models/BAAI--bge-reranker-v2-m3/` exist, set `HF_HUB_OFFLINE=1` in `.env` to make startup network-free.
3. If you edit document content directly in MongoDB (not via the API), delete `llm-models/index_cache.pkl` before restarting so BM25 reflects the new text.
4. The Colab script in `scripts/colab_bulk_embed.py` is the recommended way to do any future bulk re-embedding (e.g., switching embedding models, large data imports).

---

## Future Versions

(To be filled in as work continues)
