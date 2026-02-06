import pytest

from intelclaw.memory.embeddings import embeddings_from_config
from intelclaw.memory.session_store import SessionStore, SessionStoreConfig


@pytest.mark.asyncio
async def test_session_store_persistence_and_redaction(tmp_path):
    cfg = SessionStoreConfig(
        db_path=str(tmp_path / "sessions.db"),
        embeddings={"provider": "hash", "dimension": 64},
        redaction_mode="redact",
        exclude_last_n_messages=0,
    )

    store = SessionStore(cfg)
    await store.initialize()

    sid = await store.create_session(title="Test Session")
    await store.add_message(sid, role="user", content="hello world")
    await store.add_message(sid, role="assistant", content="hi back")

    dummy_secret = "jina_" + ("A" * 30)
    await store.add_message(sid, role="user", content=f"my key is {dummy_secret}")

    sessions = await store.list_sessions()
    assert sessions
    assert sessions[0]["session_id"] == sid

    msgs = await store.get_messages(sid)
    assert [m["role"] for m in msgs[:2]] == ["user", "assistant"]
    assert dummy_secret not in msgs[-1]["content"]
    assert "[REDACTED]" in msgs[-1]["content"]

    await store.shutdown()

    # Re-open and verify persistence
    store2 = SessionStore(cfg)
    await store2.initialize()
    msgs2 = await store2.get_messages(sid)
    assert len(msgs2) == len(msgs)
    await store2.shutdown()


@pytest.mark.asyncio
async def test_session_store_retrieval_windows(tmp_path):
    cfg = SessionStoreConfig(
        db_path=str(tmp_path / "sessions.db"),
        embeddings={"provider": "hash", "dimension": 64},
        redaction_mode="redact",
        exclude_last_n_messages=0,
        max_windows=2,
        window_messages_before=1,
        window_messages_after=1,
    )

    store = SessionStore(cfg)
    await store.initialize()

    sid1 = await store.create_session(title="Alpha")
    await store.add_message(sid1, role="user", content="I love bananas and apples.")
    await store.add_message(sid1, role="assistant", content="Bananas are great.")

    sid2 = await store.create_session(title="Beta")
    await store.add_message(sid2, role="user", content="Discuss contracts and case law.")
    await store.add_message(sid2, role="assistant", content="Sure, let's discuss the contract terms.")

    windows = await store.retrieve_context_windows(
        "bananas", session_id=sid1, max_windows=1, include_other_sessions=False, exclude_last_n_messages=0
    )
    assert windows
    assert windows[0]["session_id"] == sid1
    assert any("bananas" in m["content"].lower() for m in windows[0]["messages"])

    await store.shutdown()


def test_embeddings_fallback_to_local_hash(monkeypatch):
    monkeypatch.delenv("JINA_API_KEY", raising=False)
    emb = embeddings_from_config({"provider": "jina", "model": "jina-embeddings-v3"})
    assert getattr(emb, "name", "") == "local_hash"

