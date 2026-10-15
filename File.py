from docx import Document
import requests
import numpy as np

OLLAMA_URL = "http://localhost:11434"


# ---------------------------
# Read DOCX
# ---------------------------
def read_docx(file_path):
    doc = Document(file_path)

    text = []

    for p in doc.paragraphs:
        if p.text.strip():
            text.append(p.text)

    return "\n".join(text)


# ---------------------------
# Chunking
# ---------------------------
def chunk_text(text, chunk_size=800):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size
        chunks.append(text[start:end])
        start = end

    return chunks


# ---------------------------
# Create Embedding
# ---------------------------
def create_embedding(text):
    response = requests.post(
        f"{OLLAMA_URL}/api/embeddings",
        json={
            "model": "nomic-embed-text",
            "prompt": text
        }
    )

    return response.json()["embedding"]


# ---------------------------
# Cosine Similarity
# ---------------------------
def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)

    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


# ---------------------------
# Build Vector Store
# ---------------------------
def build_vector_store(chunks):
    store = []

    for chunk in chunks:
        embedding = create_embedding(chunk)

        store.append({
            "chunk": chunk,
            "embedding": embedding
        })

    return store


# ---------------------------
# Retrieve Top K
# ---------------------------
def retrieve(question, vector_store, top_k=3):
    question_embedding = create_embedding(question)

    results = []

    for item in vector_store:
        score = cosine_similarity(
            question_embedding,
            item["embedding"]
        )

        results.append({
            "score": score,
            "chunk": item["chunk"]
        })

    results.sort(
        key=lambda x: x["score"],
        reverse=True
    )

    return results[:top_k]


# ---------------------------
# Ask LLM
# ---------------------------
def ask_llm(question, retrieved_chunks):
    context = "\n\n".join(
        [chunk["chunk"] for chunk in retrieved_chunks]
    )

    prompt = f"""
Answer ONLY from the provided context.
CONTEXT:
{context}
QUESTION:
{question}
Provide a concise answer.
"""

    response = requests.post(
        f"{OLLAMA_URL}/api/generate",
        json={
            "model": "qwen3:8b",
            "prompt": prompt,
            "stream": False
        }
    )
    return response.json()["response"]


# ---------------------------
# Main
# ---------------------------
if __name__ == "__main__":
    doc_text = read_docx("SRS_15480_MTStar_Integration.docx")
    print(f"Document Length: {len(doc_text)}")
    chunks = chunk_text(doc_text)
    print(f"Chunks Created: {len(chunks)}")
    vector_store = build_vector_store(chunks)

    while True:

        question = input("\nAsk Question (type exit): ")

        if question.lower() == "exit":
            break

        retrieved = retrieve(
            question,
            vector_store
        )

        answer = ask_llm(
            question,
            retrieved
        )

        print("\nANSWER")
        print("------")
        print(answer)

        print("\nSOURCE CHUNKS")
        print("-------------")

        for i, chunk in enumerate(retrieved, start=1):
            print(f"\nChunk {i}")
            print(chunk["chunk"][:300])
