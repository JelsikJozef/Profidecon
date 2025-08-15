from __future__ import annotations

import time
from typing import Optional

from prometheus_client import Counter, Histogram

# Phase durations
phase_duration = Histogram(
    "profidecon_phase_duration_seconds",
    "Duration of pipeline phases in seconds",
    labelnames=("phase",),
)

# Token counts per type
pii_tokens_total = Counter(
    "profidecon_pii_tokens_total",
    "Total number of PII tokens created",
    labelnames=("type",),
)

# LLM usage
llm_calls_total = Counter(
    "profidecon_llm_calls_total",
    "Total number of LLM calls",
)
llm_latency_ms = Histogram(
    "profidecon_llm_latency_ms",
    "Latency of LLM calls in milliseconds",
)

# De-anonymization attempts
deanon_allow_total = Counter(
    "profidecon_deanon_allow_total",
    "Total allowed de-anonymization substitutions",
)

deanon_deny_total = Counter(
    "profidecon_deanon_deny_total",
    "Total denied de-anonymization substitutions",
)

# Errors per phase
errors_total = Counter(
    "profidecon_errors_total",
    "Total errors across phases",
    labelnames=("phase", "error_type"),
)

# Simple helpers
class timer:
    def __init__(self, hist: Histogram, **labels):
        self.hist = hist
        self.labels = labels
        self.t0: Optional[float] = None

    def __enter__(self):
        self.t0 = time.perf_counter()
        return self

    def __exit__(self, exc_type, exc, tb):
        dt = time.perf_counter() - (self.t0 or time.perf_counter())
        if self.labels:
            self.hist.labels(**self.labels).observe(dt)
        else:
            self.hist.observe(dt)
        return False

