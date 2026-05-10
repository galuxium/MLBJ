from __future__ import annotations

import logging

import structlog


def configure_logging(log_level: str = "INFO") -> None:
    """
    Configure structlog to emit structured JSON logs.
    All modules should obtain loggers via `structlog.get_logger()`.
    """
    level = getattr(logging, log_level.upper(), logging.INFO)

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.UnicodeDecoder(),
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )

    # Keep stdlib logging consistent (FastAPI/uvicorn use it internally)
    logging.basicConfig(format="%(message)s", level=level)
