from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI


# 1. Set up the embedding model
embeddings = GoogleGenerativeAIEmbeddings(
    model="models/gemini-embedding-001"
)


# 2. Load the text files as LangChain Documents
data_folder = Path("data/benign")

documents = []

for file_path in data_folder.glob("*.txt"):
    text = file_path.read_text(encoding="utf-8")

    document = Document(
        page_content=text,
        metadata={"source": file_path.name}
    )

    documents.append(document)


# 3. Split the documents into smaller chunks
text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=200,
    chunk_overlap=30
)

chunks = text_splitter.split_documents(documents)

print(f"Original documents: {len(documents)}")
print(f"Chunks created: {len(chunks)}")


# 4. Create a vector store using the embedding model
vector_store = InMemoryVectorStore(embeddings)


# 5. Add the chunks to the vector store
# LangChain will generate embeddings for the chunks here
vector_store.add_documents(chunks)


# 6. Ask a test question
query = "How does a language model use external information?"

#query = "How does a language model get information from documents?"
# query = "What are relational databases used for?"
# query = "Why might an organisation use cloud computing?"

# 7. Search for the two most similar chunks
results = vector_store.similarity_search(
    query,
    k=2
)


# 8. Inspect what the retriever found
print(f"\nQuery: {query}")

for i, result in enumerate(results):
    print(f"\nResult {i + 1}")
    print(f"Source: {result.metadata['source']}")
    print(result.page_content)

# 9. creating a model
llm = ChatGoogleGenerativeAI(
    model="gemini-3.5-flash-lite"
)

context = "\n\n".join(
    result.page_content for result in results
)

prompt = f"""
Answer the question using the provided context.

Context:
{context}

Question:
{query}
"""

response = llm.invoke(prompt)
print("\nLLM Answer:")
print(response.content)