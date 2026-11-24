from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import (
    GoogleGenerativeAIEmbeddings,
    ChatGoogleGenerativeAI,
)
from langchain_text_splitters import RecursiveCharacterTextSplitter


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
# 5. Set up the LLM
# --------------------------------------------------

llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)


# --------------------------------------------------
# 6. RAG function
# --------------------------------------------------

def run_rag(query):
    # Retrieve the two most relevant chunks
    results = vector_store.similarity_search(
        query,
        k=2,
    )

    # Combine retrieved chunks into one context
    context = "\n\n".join(
        result.page_content for result in results
    )

    # Construct the prompt
    prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}
"""

    # Send the query + retrieved context to Gemini
    response = llm.invoke(prompt)

    return {
        "query": query,
        "results": results,
        "response": response,
    }


# --------------------------------------------------
# 7. Manual test
# --------------------------------------------------

if __name__ == "__main__":
    query = "How does a language model use external information?"

    result = run_rag(query)

    print(f"\nQuery: {query}")

    print("\nRetrieved chunks:")

    for i, retrieved_chunk in enumerate(result["results"], start=1):
        print(f"\nResult {i}")
        print(
            f"Source: {retrieved_chunk.metadata['source']}"
        )
        print(retrieved_chunk.page_content)

    print("\nLLM Answer:")
    print(result["response"].content)