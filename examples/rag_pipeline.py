"""Retrieval-augmented generation with DakeraIndexStore and VectorStoreIndex.

Indexes documents into Dakera (the server embeds them: no local embedding
model or embedding API key), retrieves the context for a question and builds
the RAG prompt. With an LLM configured, ``index.as_query_engine()`` answers
directly.

Run a Dakera server first (see the README), then:

    pip install llamaindex-dakera
    export DAKERA_API_URL=http://localhost:3000
    export DAKERA_API_KEY=dk-mykey   # the server's root key; leave unset if auth is off
    python examples/rag_pipeline.py
"""

import os
import uuid

from llama_index.core import Document, PromptTemplate, Settings, StorageContext, VectorStoreIndex

from llama_index_dakera import DakeraIndexStore

api_url = os.environ.get("DAKERA_API_URL", "http://localhost:3000")
api_key = os.environ.get("DAKERA_API_KEY", "")

Settings.embed_model = None  # Dakera embeds on the server: no embedding API calls

store = DakeraIndexStore(
    api_url=api_url,
    api_key=api_key,
    namespace=f"llamaindex-rag-{uuid.uuid4().hex[:8]}",
)
documents = [
    Document(
        text="Dakera is an AI agent memory server with server-side embedding.",
        metadata={"topic": "overview"},
        id_="overview",
    ),
    Document(
        text="Dakera decays memory importance over time, weighted by access.",
        metadata={"topic": "concepts"},
        id_="decay",
    ),
    Document(
        text="LlamaIndex orchestrates data ingestion and retrieval for LLMs.",
        metadata={"topic": "framework"},
        id_="llamaindex",
    ),
]
index = VectorStoreIndex.from_documents(
    documents, storage_context=StorageContext.from_defaults(vector_store=store)
)
print(f"Indexed {len(documents)} documents.")

question = "How does Dakera handle memory decay?"
nodes = index.as_retriever(similarity_top_k=2).retrieve(question)
print(f"\nRetrieved for {question!r}:")
for hit in nodes:
    print(f"  [{hit.score:.3f}] ({hit.node.ref_doc_id}) {hit.node.get_content()}")

prompt = PromptTemplate(
    "Answer from the context only.\n\nContext:\n{context}\n\nQuestion: {question}"
)
context = "\n".join(f"- {hit.node.get_content()}" for hit in nodes)
print("\nRAG prompt:\n" + prompt.format(context=context, question=question))
# With an LLM (e.g. Settings.llm = OpenAI(model="gpt-4o")):
#     print(index.as_query_engine(similarity_top_k=2).query(question))

index.delete_ref_doc("llamaindex")  # removes the document's nodes from Dakera
print("\nDeleted the 'llamaindex' document.")
