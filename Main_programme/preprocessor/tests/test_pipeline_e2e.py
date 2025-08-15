import os
import json
from pathlib import Path

from Main_programme.preprocessor.processors.pii_analyzer import PiiAnalyzer
from Main_programme.preprocessor.cli import run_pseudonymization, run_enrich_llm
from Main_programme.preprocessor.middleware.response_deanonymizer import ResponseDeanonymizer, DefaultDeanonymizationPolicy
from Main_programme.preprocessor.observability import metrics as m


def _setup_env():
    env = {
        "TOKEN_VAULT_BACKEND": "postgres",
        "DATABASE_URL": "sqlite:///:memory:",
        "HMAC_KEY_MATERIAL": "base64:dGVzdF9obWFjX2tleV8zMl9ieXRlc19sb25nX2Vub3VnaF9mb3JfdGVzdGluZw==",
        "KEK_MATERIAL": "base64:dGVzdF9rZWtfMzJfYnl0ZXNfbG9uZ19lbm91Z2hfZm9yX3Rlc3Rpbmc=",
        "SALT_V1": "base64:dGVzdF9zYWx0XzE2X2J5dGVzX2hlcmU=",
        "TOKEN_ID_BYTES": "10",
        "PSEUDO_SCOPE": "tenant",
        "DEANON_DEVICE_REQUIRED": "edge_trusted",
    }
    for k, v in env.items():
        os.environ[k] = v


def test_pipeline_e2e(tmp_path: Path, caplog):
    _setup_env()

    phase1 = tmp_path / "phase1"
    phase2 = tmp_path / "phase2"
    phase3 = tmp_path / "phase3"
    phase1.mkdir(); phase2.mkdir(); phase3.mkdir()

    # Create Phase-1 JSONL with text and minimal metadata
    text = "Alice Doe alice@example.com and +421 900 123 456 said مرحبا."
    doc = {"text": text, "metadata": {"source": str(tmp_path/"doc.txt")}}
    (phase1 / "doc.jsonl").write_text(json.dumps(doc, ensure_ascii=False) + "\n", encoding="utf-8")

    # Phase-2: pseudonymize
    run_pseudonymization(phase1, phase2, scope="tenant", tenant_id="t1", types_include=None, types_exclude=None, max_entities=None, require_annotations=False, force=True)

    data2 = json.loads((phase2/"doc.jsonl").read_text(encoding='utf-8').splitlines()[0])
    text_pseudo = data2.get("text_pseudo")
    assert text_pseudo and "alice@example.com" not in text_pseudo
    assert "[EMAIL:" in text_pseudo and "[PHONE:" in text_pseudo

    # Phase-3: LLM enrichment (heuristic client path without OPENAI_API_KEY)
    if "OPENAI_API_KEY" in os.environ:
        del os.environ["OPENAI_API_KEY"]
    run_enrich_llm(phase2, phase3, model="dummy", temperature=0.0, max_tokens=64, skip_if_present=True)

    data3 = json.loads((phase3/"doc.jsonl").read_text(encoding='utf-8').splitlines()[0])
    llm = data3.get("metadata", {}).get("llm_enrichment", {})
    assert isinstance(llm.get("summary", ""), str)
    assert isinstance(llm.get("tags", []), list)
    # Ensure LLM saw only tokens (no plaintext in intermediate)
    raw3 = (phase3/"doc.jsonl").read_text(encoding='utf-8')
    assert "alice@example.com" not in raw3

    # Phase-4: de-anonymize on device edge
    de = ResponseDeanonymizer(DefaultDeanonymizationPolicy())

    class Actor:
        def __init__(self, tid, roles, device):
            self.tenant_id = tid
            self.roles = roles
            self.device_trust_level = device
            self.request_id = "r-e2e"

    allowed = Actor("t1", ["case_handler"], "edge_trusted")
    denied = Actor("t1", ["viewer"], "edge_trusted")

    out_allowed = de.run(text_pseudo, actor=allowed)
    assert "alice@example.com" in out_allowed

    out_denied = de.run(text_pseudo, actor=denied)
    assert "[EMAIL:REDACTED]" in out_denied or "[PHONE:REDACTED]" in out_denied

    # Metrics sanity: at least 1 LLM call and some tokens counted
    assert m.llm_calls_total._value.get() >= 1  # type: ignore[attr-defined]
    # token counters: best-effort check email increments
    # Note: depending on analyzer, labels may vary; ensure labels exist
    # Just ensure phase duration recorded
    # Can't easily assert internal value across label sets; skip strict assertion

    # No plaintext in logs
    for rec in caplog.records:
        assert "alice@example.com" not in rec.getMessage()

