from intelclaw.memory.tool_docs import ToolDocsIndex


def test_tool_docs_parser_extracts_signatures_and_description() -> None:
    md = """# TOOLS

## RAG

### Indexing
```
rag_index_path(path: str, title: str | None = None) -> RagIndexResult
rag_list_documents() -> List[DocumentInfo]
```
Use these to index PDFs and list indexed docs.

### Deletion
```
rag_delete_document(doc_id: str, confirm: bool = False) -> Result
```
Requires confirm=true.
"""

    idx = ToolDocsIndex.parse_markdown(md)
    names = idx.list_tool_names()

    assert "rag_index_path" in names
    assert "rag_list_documents" in names
    assert "rag_delete_document" in names

    hits = idx.search("index pdf", top_k=3)
    assert hits
    assert hits[0].tool_name == "rag_index_path"

    delete_hits = idx.search("confirm delete", top_k=3)
    assert any(h.tool_name == "rag_delete_document" for h in delete_hits)

