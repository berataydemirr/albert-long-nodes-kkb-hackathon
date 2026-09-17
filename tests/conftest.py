import pytest


@pytest.fixture(autouse=True)
def isolated_evidence_store(monkeypatch, tmp_path):
    from src.agent import evidence
    monkeypatch.setattr(evidence, "STORE_PATH", tmp_path / "evidence.sqlite3")
