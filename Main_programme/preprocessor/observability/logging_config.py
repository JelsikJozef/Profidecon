from __future__ import annotations

import json
import logging
import os
import re
from typing import Any, Dict

SAFE_TOKEN_RE = re.compile(r"\[([A-Z_]+):([0-9A-Z]+)\]")


class JsonRedactingFormatter(logging.Formatter):
    def __init__(self, phase: str | None = None):
        super().__init__()
        self.phase = phase

    def format(self, record: logging.LogRecord) -> str:
        payload: Dict[str, Any] = {
            "timestamp": self.formatTime(record),
            "level": record.levelname,
            "logger": record.name,
            "message": record.getMessage(),
        }
        if self.phase:
            payload["phase"] = self.phase
        # Never log plaintext PII; allow bracket tokens only
        msg = payload.get("message", "")
        # Optionally, we could mask obvious emails/phones
        msg = re.sub(r"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}", "<redacted_email>", msg)
        msg = re.sub(r"\+?[0-9][0-9\s\-]{5,}", "<redacted_phone>", msg)
        # Keep bracketed tokens as-is (pseudonymized)
        payload["message"] = msg
        return json.dumps(payload, ensure_ascii=False)


def setup_json_logging(phase: str | None = None) -> None:
    if os.getenv("PROFIDECON_JSON_LOGS", "true").lower() != "true":
        return
    handler = logging.StreamHandler()
    handler.setFormatter(JsonRedactingFormatter(phase=phase))
    root = logging.getLogger()
    root.handlers = [handler]
    # Keep level as-is; caller sets level

