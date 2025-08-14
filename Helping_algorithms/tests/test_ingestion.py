import pytest
from pathlib import Path
from Main_programme.preprocessor import ingest_batch
from Main_programme.preprocessor.models import RawDocument


@pytest.mark.asyncio
async def test_ingest_batch(tmp_path: Path):
    # príprava 3 súborov
    f1 = tmp_path / "a.pdf";
    f1.write_text("dummy")
    f2 = tmp_path / "b.docx";
    f2.write_text("dummy")
    f3 = tmp_path / "c.txt";
    f3.write_text("skip")

    docs = await ingest_batch(tmp_path)
    paths = {d.path.name for d in docs}

    assert "a.pdf" in paths
    assert "b.docx" in paths
    assert "c.txt" not in paths
    assert all(isinstance(d, RawDocument) for d in docs)
