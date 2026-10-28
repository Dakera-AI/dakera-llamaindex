"""Semantic, hybrid and BM25 retrieval with DakeraIndexStore.

The Dakera server embeds the documents (no local embedding model) and also
indexes them for BM25. LlamaIndex retrievers pick the mode: ``default``
(semantic), ``hybrid`` (semantic + BM25 weighted by ``alpha``) or
``text_search`` (BM25), with metadata filters.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/hybrid_search.py
"""

import os
import uuid

from llama_index.core import Document, Settings, StorageContext, VectorStoreIndex
from llama_index.core.base.base_retriever import BaseRetriever
from llama_index.core.vector_stores.types import FilterOperator, MetadataFilter, MetadataFilters

from llama_index_dakera import DakeraIndexStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

Settings.embed_model = None  # Dakera embeds on the server: no embedding API calls

store = DakeraIndexStore(
    api_url=api_url,
    api_key=api_key,
    namespace=f"llamaindex-hybrid-{uuid.uuid4().hex[:8]}",
)
documents = [
    Document(
        text="Python is a high-level language known for readability.", metadata={"year": 1991}
    ),
    Document(
        text="Rust provides memory safety without garbage collection.", metadata={"year": 2015}
    ),
    Document(text="TypeScript adds static types to JavaScript.", metadata={"year": 2012}),
    Document(
        text="Go is designed for concurrent systems programming at Google.", metadata={"year": 2009}
    ),
    Document(
        text="FastAPI is a modern Python web framework built on Starlette.", metadata={"year": 2018}
    ),
]
index = VectorStoreIndex.from_documents(
    documents, storage_context=StorageContext.from_defaults(vector_store=store)
)
print(f"Indexed {len(documents)} documents.")


def show(title: str, query: str, retriever: BaseRetriever) -> None:
    print(f"\n--- {title}: {query!r} ---")
    for hit in retriever.retrieve(query):
        print(f"  [{hit.score:.3f}] {hit.node.get_content()}")


show("Semantic", "memory safe language", index.as_retriever(similarity_top_k=2))
show(
    "Hybrid, alpha=0.3",
    "Starlette",
    index.as_retriever(similarity_top_k=2, vector_store_query_mode="hybrid", alpha=0.3),
)
show(
    "BM25",
    "Google concurrent",
    index.as_retriever(similarity_top_k=1, vector_store_query_mode="text_search"),
)
newer = MetadataFilters(
    filters=[MetadataFilter(key="year", value=2010, operator=FilterOperator.GT)]
)
show("Filtered, year > 2010", "programming", index.as_retriever(similarity_top_k=3, filters=newer))
