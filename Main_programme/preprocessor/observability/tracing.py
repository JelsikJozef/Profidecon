from __future__ import annotations

import contextlib
from typing import Optional

try:
    from opentelemetry import trace  # type: ignore
    _otel_available = True
except Exception:  # pragma: no cover
    _otel_available = False


@contextlib.contextmanager
def span(name: str, **attrs):
    """Create a tracing span if OpenTelemetry is available, else no-op."""
    if _otel_available:
        tracer = trace.get_tracer("profidecon")
        with tracer.start_as_current_span(name) as sp:
            for k, v in attrs.items():
                try:
                    sp.set_attribute(k, v)
                except Exception:
                    pass
            yield sp
    else:
        yield None

