from pathlib import Path

from intelclaw.contacts.store import ContactsStore


def test_contacts_store_upsert_and_lookup(tmp_path: Path):
    path = tmp_path / "contacts.md"
    store = ContactsStore(path)

    store.upsert(name="Alice", phone="+92 317-115-6353", inbound_allowed=True, notes="friend")
    store.upsert(name="Bob", phone="12345", inbound_allowed=False)

    all_contacts = store.load()
    assert len(all_contacts) == 2

    alice = store.lookup("alice")[0]
    assert alice.phone == "923171156353"
    assert alice.inbound_allowed is True

    inbound_only = store.lookup("9", inbound_only=True)
    assert len(inbound_only) == 1
    assert inbound_only[0].name == "Alice"

