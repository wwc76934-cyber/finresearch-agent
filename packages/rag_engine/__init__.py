"""金融 Agentic RAG: structure-aware chunking, hybrid retrieval, routing, self-RAG."""
from .agentic import agentic_search, route
from .index import build_chunks, get_retriever, persist_chunks
from .retrieval import Retriever

__all__ = ["agentic_search", "route", "build_chunks", "get_retriever", "persist_chunks", "Retriever"]
