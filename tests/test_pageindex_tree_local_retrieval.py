import pytest

from intelclaw.memory.agentic_rag import AgenticRAG


@pytest.mark.asyncio
async def test_pageindex_tree_local_retrieval_scores_expected_node(tmp_path) -> None:
    rag = AgenticRAG(user_id="u1", persist_dir=str(tmp_path / "rag"))
    await rag.initialize({"pageindex": {"max_nodes_per_doc": 5}})

    payload = {
        "tree": {
            "nodes": [
                {
                    "id": "n1",
                    "title": "Indemnification",
                    "summary": "Indemnity clause details.",
                    "text": "The indemnification clause covers losses and liabilities.",
                    "page_index": 3,
                    "children": [],
                },
                {
                    "id": "n2",
                    "title": "Force Majeure",
                    "summary": "Force majeure clause details.",
                    "text": "The force majeure clause excuses performance for events beyond control.",
                    "page_index": 5,
                    "children": [],
                },
            ]
        }
    }

    tree = rag._pageindex_tree_to_document_tree(
        doc_id="doc123",
        title="contract.pdf",
        tree_payload=payload,
        local_path="C:/docs/contract.pdf",
        meta={"pageNum": 10, "name": "contract.pdf"},
    )

    rag.document_trees["doc123"] = tree

    results = await rag._tree_retrieve("force majeure clause", doc_types=["pdf"])
    assert results
    assert results[0]["doc_id"] == "doc123"
    assert results[0]["node_id"] == "n2"
    assert "force majeure" in results[0]["title"].lower()

