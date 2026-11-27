from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter

from defenses.context_separation import build_context_separated_prompt
from defenses.content_filter import filter_chunks


# --------------------------------------------------
# 1. Embedding model
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# --------------------------------------------------
# 2. Load benign and adversarial documents
# --------------------------------------------------

data_folders = [
    Path("data/benign"),
    Path("data/adversarial"),
]

documents = []

for data_folder in data_folders:
    for file_path in data_folder.glob("*.txt"):
        text = file_path.read_text(encoding="utf-8")

        document = Document(
            page_content=text,
            metadata={"source": file_path.name},
        )

        documents.append(document)


# --------------------------------------------------
# 3. Split documents into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)

chunks = text_splitter.split_documents(documents)


# --------------------------------------------------
# 4. Create vector store
# --------------------------------------------------

vector_store = InMemoryVectorStore(embeddings)

vector_store.add_documents(chunks)


# --------------------------------------------------
# 5. Set up LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# --------------------------------------------------
# 6. RAG function
# --------------------------------------------------

def run_rag(query, defense="none"):

    # First retrieve the top 2 relevant chunks.
    results = vector_store.similarity_search(
        query,
        k=2,
    )

    blocked_chunks = []

    # --------------------------------------------------
    # Optional content-filter defence
    # --------------------------------------------------

    if defense == "content_filtering":

        print("\n--- FILTER INPUT ---")

        for chunk in results:
            print(f"\nSource: {chunk.metadata['source']}")
            print(repr(chunk.page_content))

        allowed_chunks, blocked_chunks = filter_chunks(results)

        # After filtering, only allowed chunks should continue
        # into the LLM context.
        results = allowed_chunks

        if not results:
            print(
                "\nAll retrieved chunks were blocked "
                "by content filtering."
            )

            return {
                "query": query,
                "results": [],
                "blocked_chunks": blocked_chunks,
                "response": None,
            }

    # --------------------------------------------------
    # Construct context from allowed chunks
    # --------------------------------------------------

    context = "\n\n".join(
        result.page_content
        for result in results
    )


    # --------------------------------------------------
    # Select prompt strategy
    # --------------------------------------------------

    if defense == "context_separation":

        prompt = build_context_separated_prompt(
            query=query,
            context=context,
        )

    else:

        prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}
"""


    # --------------------------------------------------
    # Send context + question to Gemini
    # --------------------------------------------------

    response = llm.invoke(prompt)


    return {
        "query": query,
        "results": results,
        "blocked_chunks": blocked_chunks,
        "response": response,
    }


# --------------------------------------------------
# 7. Manual content-filter test
# --------------------------------------------------

if __name__ == "__main__":

    query = "How does a language model use external information?"

    result = run_rag(
        query,
        defense="content_filtering",
    )

    print(f"\nQuery: {query}")


    # --------------------------------------------------
    # Allowed chunks
    # --------------------------------------------------

    print("\nAllowed chunks:")

    for i, retrieved_chunk in enumerate(
        result["results"],
        start=1,
    ):
        print(f"\nResult {i}")
        print(
            f"Source: "
            f"{retrieved_chunk.metadata['source']}"
        )
        print(retrieved_chunk.page_content)


    # --------------------------------------------------
    # Blocked chunks
    # --------------------------------------------------

    print("\nBlocked chunks:")

    if not result["blocked_chunks"]:
        print("None")

    for blocked in result["blocked_chunks"]:

        chunk = blocked["chunk"]

        print(f"\nSource: {chunk.metadata['source']}")

        print(
            f"Matched patterns: "
            f"{blocked['matched_patterns']}"
        )

        print(chunk.page_content)


    # --------------------------------------------------
    # LLM answer
    # --------------------------------------------------

    print("\nLLM Answer:")

    if result["response"] is not None:
        print(result["response"].content)

    else:
        print(
            "No response generated because all "
            "retrieved chunks were blocked."
        )