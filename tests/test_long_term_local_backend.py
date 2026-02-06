import pytest

from intelclaw.memory.long_term import LongTermMemory


@pytest.mark.asyncio
async def test_long_term_local_sqlite_persistence_and_search(tmp_path) -> None:
    cfg = {
        "long_term": {"backend": "local", "db_path": str(tmp_path / "ltm.db")},
        "redaction": {"on_detect": "skip"},
    }
    ltm = LongTermMemory(user_id="u1", config=cfg)
    await ltm.initialize()

    mem_id = await ltm.add(
        "User preference: use Python for scripts",
        metadata={"kind": "user_preference", "importance": 0.9},
    )
    assert mem_id

    hits = await ltm.search("python scripts", limit=5)
    assert any("use Python" in h.get("content", "") for h in hits)

    await ltm.shutdown()


@pytest.mark.asyncio
async def test_long_term_skips_secret_like_content_by_default(tmp_path) -> None:
    cfg = {
        "long_term": {"backend": "local", "db_path": str(tmp_path / "ltm.db")},
        "redaction": {"on_detect": "skip"},
    }
    ltm = LongTermMemory(user_id="u1", config=cfg)
    await ltm.initialize()

    secretish = "here is the api key: 0123456789abcdef0123456789abcdef"
    mem_id = await ltm.add(secretish, metadata={"kind": "fact"})
    assert mem_id == ""

    all_mems = await ltm.get_all()
    assert all_mems == []

    await ltm.shutdown()


@pytest.mark.asyncio
async def test_long_term_redacts_when_configured(tmp_path) -> None:
    cfg = {
        "long_term": {"backend": "local", "db_path": str(tmp_path / "ltm.db")},
        "redaction": {"on_detect": "redact"},
    }
    ltm = LongTermMemory(user_id="u1", config=cfg)
    await ltm.initialize()

    mem_id = await ltm.add("PAGEINDEX_API_KEY=not-a-real-secret", metadata={"kind": "fact"})
    assert mem_id

    all_mems = await ltm.get_all()
    assert all_mems
    assert "PAGEINDEX_API_KEY=" in all_mems[0]["content"]
    assert "not-a-real-secret" not in all_mems[0]["content"]
    assert "[REDACTED]" in all_mems[0]["content"]

    await ltm.shutdown()
