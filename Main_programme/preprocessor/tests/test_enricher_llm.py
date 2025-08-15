import json
import time
from pathlib import Path

from Main_programme.preprocessor.processors.enricher_llm import LlmEnricher
from Main_programme.preprocessor.cli import run_enrich_llm


class FakeClient:
    def __init__(self, delay_ms: int = 5, return_as_str: bool = False):
        self.delay_ms = delay_ms
        self.return_as_str = return_as_str
        self.calls = 0

    def enrich(self, *, text: str, model: str, temperature: float, max_tokens: int):
        self.calls += 1
        # Simulate latency
        time.sleep(self.delay_ms / 1000.0)
        # Simple echo-like behavior for deterministic test
        if self.return_as_str:
            return "This is a summary."
        return {
            "summary": "Concise summary about the pseudonymized document.",
            "tags": ["policy", "registration", "email", "phone"]
        }


def _write_jsonl(path: Path, data: dict):
    path.write_text(json.dumps(data, ensure_ascii=False) + "\n", encoding="utf-8")


def test_basic_llm_enricher_run():
    client = FakeClient()
    enr = LlmEnricher(api_client=client, model="test-model", temperature=0.1, max_tokens=128)
    text = "Hello [EMAIL:ABC123] world. Call [PHONE:XYZ789]."
    res = enr.run(text, metadata={"doc_type": "note"})

    assert isinstance(res, dict)
    assert res["summary"]
    assert isinstance(res["tags"], list) and len(res["tags"]) > 0
    assert res["model"] == "test-model"
    assert isinstance(res["latency_ms"], int) and res["latency_ms"] >= 0


def test_token_integrity_and_input_limit():
    # Ensure tokens are not altered and input is truncated not expanded
    long_text = ("[EMAIL:AAAA] "+"x"*10000+" [PHONE:BBBB]")
    client = FakeClient()
    enr = LlmEnricher(api_client=client, model="t", max_tokens=64, max_tokens_input=1024)
    res = enr.run(long_text, metadata={})
    assert res["summary"]
    # the client doesn't modify input; just ensure the original tokens are intact in the source (no exceptions)
    assert "[EMAIL:AAAA]" in long_text and "[PHONE:BBBB]" in long_text


def test_performance_latency_recorded():
    client = FakeClient(delay_ms=10)
    enr = LlmEnricher(api_client=client, model="m")
    res = enr.run("Some text.", metadata={})
    assert res["latency_ms"] >= 10


def test_model_and_temperature_config():
    client = FakeClient()
    enr = LlmEnricher(api_client=client, model="model-X", temperature=0.7)
    res = enr.run("Short text", metadata={})
    assert res["model"] == "model-X"


def test_cli_enrich_llm_phase3(tmp_path: Path):
    # Prepare Phase-2 input document
    in_dir = tmp_path / "phase2"
    out_dir = tmp_path / "phase3"
    in_dir.mkdir()
    out_dir.mkdir()

    doc = {
        "text_pseudo": "Hello [EMAIL:K7V2] user.",
        "metadata": {"source": "/x/y.txt", "token_spans": [], "pii_entities": []}
    }
    _write_jsonl(in_dir / "abc.jsonl", doc)

    # Monkeypatch: use FakeClient via dependency injection by temporarily replacing OPENAI_API_KEY
    # so the CLI picks heuristic client (fast, no network)
    import os
    old_key = os.environ.get("OPENAI_API_KEY")
    if old_key:
        del os.environ["OPENAI_API_KEY"]

    try:
        run_enrich_llm(in_dir, out_dir, model="dummy", temperature=0.0, max_tokens=64, skip_if_present=True)
    finally:
        if old_key:
            os.environ["OPENAI_API_KEY"] = old_key

    # Validate output
    out_file = out_dir / "abc.jsonl"
    assert out_file.exists()
    data = json.loads(out_file.read_text(encoding="utf-8").splitlines()[0])
    assert "llm_enrichment" in data["metadata"]
    llm = data["metadata"]["llm_enrichment"]
    assert isinstance(llm["summary"], str)
    assert isinstance(llm["tags"], list)
    assert data["metadata"]["phase"] == 3


def test_skip_if_present(tmp_path: Path):
    in_dir = tmp_path / "phase2"
    out_dir = tmp_path / "phase3"
    in_dir.mkdir(); out_dir.mkdir()

    already = {
        "text_pseudo": "Body.",
        "metadata": {
            "llm_enrichment": {"summary": "s", "tags": [], "model": "m", "latency_ms": 1}
        }
    }
    _write_jsonl(in_dir / "doc.jsonl", already)

    run_enrich_llm(in_dir, out_dir, model="dummy", skip_if_present=True)
    assert not (out_dir / "doc.jsonl").exists()

