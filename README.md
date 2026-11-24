# RAG Prompt Injection Security

This project evaluates defences against indirect prompt injection in Retrieval-Augmented Generation (RAG) systems.

The project compares two defence strategies, **context separation** and **content filtering**, both individually and together, against an undefended baseline.

## Current Implementation

A basic RAG pipeline has been implemented:

**Documents → Chunking → Embeddings → Vector Store → Retrieval → LLM → Answer**

The current implementation loads local documents, generates Gemini embeddings, performs similarity-based retrieval, and provides the retrieved context to a Gemini language model for answer generation.

## Planned Experiments

Four configurations will be evaluated:

- No defence
- Context separation
- Content filtering
- Context separation + content filtering

The next phase will introduce controlled indirect prompt-injection examples and establish baseline attack behaviour.
