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
# 1. Data folders
# --------------------------------------------------

BENIGN_FOLDER = Path("data/benign")
ADVERSARIAL_FOLDER = Path("data/adversarial")


# --------------------------------------------------
# 2. Embedding model
# --------------------------------------------------

embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# --------------------------------------------------
# 3. Text splitter
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=300,
    chunk_overlap=30,
)


# --------------------------------------------------
# 4. LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# --------------------------------------------------
# 5. Load one text file
# --------------------------------------------------

def load_document(file_path):
    text = file_path.read_text(
        encoding="utf-8"
    )

    return Document(
        page_content=text,
        metadata={
            "source": file_path.name
        },
    )


# --------------------------------------------------
# 6. Build corpus for ONE experiment
# --------------------------------------------------

def build_vector_store(attack_file=None):
    """
    Build a fresh vector store containing:

    - all benign documents
    - only the attack document for the current test

    This prevents different attack files from competing
    against one another during retrieval.
    """

    documents = []

    # Load all benign documents
    for file_path in sorted(
        BENIGN_FOLDER.glob("*.txt")
    ):
        documents.append(
            load_document(file_path)
        )

    # Load only this experiment's attack file
    if attack_file is not None:
        attack_path = (
            ADVERSARIAL_FOLDER
            / attack_file
        )

        if not attack_path.exists():
            raise FileNotFoundError(
                f"Attack file not found: {attack_path}"
            )

        documents.append(
            load_document(attack_path)
        )

    # Split documents using frozen experiment settings
    chunks = text_splitter.split_documents(
        documents
    )

    # Create a fresh vector store for this run
    vector_store = InMemoryVectorStore(
        embeddings
    )

    vector_store.add_documents(
        chunks
    )

    return vector_store


# --------------------------------------------------
# 7. Run one RAG experiment
# --------------------------------------------------

def run_rag(
    query,
    defense="none",
    attack_file=None,
):
    """
    Supported defence modes:

    none
        No defence.

    context_separation
        Strong context-separation prompt.

    content_filtering
        Rule-based filtering before generation.

    combined
        Content filtering followed by strong
        context separation.
    """

    valid_defenses = {
        "none",
        "context_separation",
        "content_filtering",
        "combined",
    }

    if defense not in valid_defenses:
        raise ValueError(
            f"Unknown defense: {defense}"
        )

    # ----------------------------------------------
    # Build isolated corpus for this test
    # ----------------------------------------------

    vector_store = build_vector_store(
        attack_file=attack_file
    )

    # ----------------------------------------------
    # Retrieve top 2 relevant chunks
    # ----------------------------------------------

    retrieved_chunks = (
        vector_store.similarity_search(
            query,
            k=2,
        )
    )

    # Preserve retrieval exactly as it happened
    # BEFORE any defence changes the chunks.
    original_results = list(
        retrieved_chunks
    )

    blocked_chunks = []

    # ----------------------------------------------
    # Content filtering
    # ----------------------------------------------

    if defense in {
        "content_filtering",
        "combined",
    }:
        (
            allowed_chunks,
            blocked_chunks,
        ) = filter_chunks(
            retrieved_chunks
        )

        retrieved_chunks = allowed_chunks

    # ----------------------------------------------
    # If filter blocked everything
    # ----------------------------------------------

    if not retrieved_chunks:
        return {
            "query": query,
            "defense": defense,
            "attack_file": attack_file,
            "original_results": original_results,
            "results": [],
            "blocked_chunks": blocked_chunks,
            "response": None,
        }

    # ----------------------------------------------
    # Context contains ONLY chunks that survived
    # any filtering.
    # ----------------------------------------------

    context = "\n\n".join(
        chunk.page_content
        for chunk in retrieved_chunks
    )

    # ----------------------------------------------
    # Prompt selection
    # ----------------------------------------------

    if defense in {
        "context_separation",
        "combined",
    }:
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

    # ----------------------------------------------
    # Send to Gemini
    # ----------------------------------------------

    response = llm.invoke(
        prompt
    )

    return {
        "query": query,
        "defense": defense,
        "attack_file": attack_file,
        "original_results": original_results,
        "results": retrieved_chunks,
        "blocked_chunks": blocked_chunks,
        "response": response,
    }


# --------------------------------------------------
# 8. Manual sanity test
# --------------------------------------------------

if __name__ == "__main__":
    query = (
        "How does a language model "
        "use external information?"
    )

    result = run_rag(
        query=query,
        defense="combined",
        attack_file="attack_01.txt",
    )

    print(f"\nQuery: {result['query']}")
    print(f"Defense: {result['defense']}")
    print(
        f"Attack file: "
        f"{result['attack_file']}"
    )

    print("\nOriginally retrieved:")

    for index, chunk in enumerate(
        result["original_results"],
        start=1,
    ):
        print(f"\nResult {index}")
        print(
            f"Source: "
            f"{chunk.metadata['source']}"
        )
        print(chunk.page_content)

    print("\nAllowed after filtering:")

    if not result["results"]:
        print("None")

    for index, chunk in enumerate(
        result["results"],
        start=1,
    ):
        print(f"\nResult {index}")
        print(
            f"Source: "
            f"{chunk.metadata['source']}"
        )
        print(chunk.page_content)

    print("\nBlocked by filter:")

    if not result["blocked_chunks"]:
        print("None")

    for blocked in result["blocked_chunks"]:
        chunk = blocked["chunk"]

        print(
            f"\nSource: "
            f"{chunk.metadata['source']}"
        )

        print(
            f"Matched patterns: "
            f"{blocked['matched_patterns']}"
        )

        print(chunk.page_content)

    print("\nLLM Answer:")

    if result["response"] is None:
        print(
            "No response generated because all "
            "retrieved chunks were blocked."
        )
    else:
        print(result["response"].content)