Yes. And for this demo, I would deliberately keep it simple but not toy-like:

> One .docx → extract text → chunk it → Ollama creates embeddings → ChromaDB stores vectors + text + metadata → user asks a question → Ollama embeds the question → Chroma retrieves the most relevant chunks → Ollama generates a grounded answer.



Ollama's current embedding API supports batching multiple inputs, and its documentation recommends using the same embedding model for indexing and querying.  Chroma can store embeddings, documents and metadata and perform nearest-neighbor queries. 

Below is a proper demo document you can keep as your first RAG implementation reference.

Demo: Local RAG with Python + Ollama + ChromaDB

1. What we are building

Word document
                      │
                      ▼
              ┌───────────────┐
              │ python-docx   │
              │ text extraction│
              └───────┬───────┘
                      │
                      ▼
                  Chunking
                      │
                      ▼
              ┌───────────────┐
              │    Ollama     │
              │  Embedding    │
              │     model     │
              └───────┬───────┘
                      │
                      ▼
              ┌───────────────┐
              │   ChromaDB    │
              │               │
              │ vectors       │
              │ text          │
              │ metadata      │
              └───────┬───────┘
                      │
                 User question
                      │
                      ▼
              Query embedding
                      │
                      ▼
              Chroma similarity
                  search
                      │
                      ▼
             Top relevant chunks
                      │
                      ▼
              ┌───────────────┐
              │    Ollama     │
              │   LLM model   │
              └───────┬───────┘
                      │
                      ▼
              Grounded answer
                 + sources

There are actually two different Ollama jobs here:

1. Embedding model → converts text into vectors.


2. Generative LLM → reads the retrieved text and writes the answer.



Don't confuse those two.


---

2. Why do we need chunking?

Suppose your Word document contains:

Introduction
...
Architecture
...
Problem
...
Root Cause
...
Resolution
...
Lessons Learned
...

We don't want to put the entire document into one vector.

Instead:

Document
   │
   ├── Chunk 1
   ├── Chunk 2
   ├── Chunk 3
   ├── Chunk 4
   └── Chunk 5

Why?

Because a user may ask:

> "What was the root cause?"



Only one or two sections may be relevant.

If the entire document is one vector, the representation becomes too coarse.


---

3. Install the software

You need:

Python

Ollama

an Ollama generative model

an Ollama embedding model

ChromaDB

python-docx


Install Python dependencies:

pip install ollama chromadb python-docx

Chroma's current Python quickstart uses pip install chromadb. 

Then install/run Ollama.

For embeddings, Ollama currently recommends models such as embeddinggemma, qwen3-embedding, and all-minilm. 

For this demo, use:

ollama pull embeddinggemma

And choose a normal chat model available on your machine, for example:

ollama pull gemma3

The exact generative model isn't important for understanding RAG.


---

4. Project structure

Start with:

rag-demo/
│
├── data/
│   └── sample.docx
│
├── chroma_db/
│
├── ingest.py
├── ask.py
└── requirements.txt

ingest.py puts knowledge into the vector database.

ask.py retrieves knowledge and asks the LLM to answer.

This separation is intentional.

Ingestion is normally done once, while querying happens many times.


---

5. Step 1 — Read the Word document

python-docx provides Document() for opening a .docx, and Word documents expose paragraphs and tables programmatically. 

Start simple:

from docx import Document


def extract_text(file_path):
    document = Document(file_path)

    parts = []

    # Normal paragraphs
    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    # Tables
    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]

            row_text = " | ".join(
                cell for cell in cells if cell
            )

            if row_text:
                parts.append(row_text)

    return "\n".join(parts)


text = extract_text("data/sample.docx")

print(text)

What happens?

sample.docx
     ↓
python-docx
     ↓
paragraphs + tables
     ↓
plain text

Important limitation

A Word document isn't necessarily just paragraphs.

It can contain:

paragraphs

tables

headers

footers

images

text boxes

embedded objects


The basic demo above handles paragraphs and ordinary tables. python-docx also exposes document-order iteration through iter_inner_content(), which can be useful when you need paragraphs and tables in their original order. 

For your first demo, don't try to solve every possible Word format.


---

6. Step 2 — Chunk the document

A very basic chunker:

def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks

Then:

chunks = chunk_text(text)

for i, chunk in enumerate(chunks):
    print(f"\n--- CHUNK {i} ---")
    print(chunk)


---

7. Why overlap?

Imagine the document contains:

Chunk 1:
The payment service experienced...

Chunk 2:
...the root cause was connection pool exhaustion...

The important sentence could get split across boundaries.

Overlap gives neighboring chunks some shared context.

Chunk 1
████████████████████
              ████████████████████
              Chunk 2

For the demo:

chunk_size = 1000
overlap = 150

is a reasonable starting point.

Don't assume these are universally optimal values.

Later, you'll experiment based on your actual documents.


---

8. Step 3 — Generate embeddings with Ollama

This is where AI starts entering the pipeline.

import ollama


EMBEDDING_MODEL = "embeddinggemma"


def create_embeddings(chunks):
    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    return response["embeddings"]

Notice something important:

input=chunks

We're sending multiple chunks in one embedding request.

Ollama's embedding API supports an array of input strings for batch embedding, which is useful for reducing per-request overhead. 

Then:

embeddings = create_embeddings(chunks)

print("Chunks:", len(chunks))
print("Embeddings:", len(embeddings))
print("Vector dimensions:", len(embeddings[0]))

Conceptually:

Chunk 1 → [0.01, -0.23, 0.17, ...]
Chunk 2 → [0.09, -0.12, 0.43, ...]
Chunk 3 → [0.02, -0.44, 0.21, ...]

These vectors represent the semantic meaning of the chunks.


---

9. Step 4 — Store them in ChromaDB

Create a persistent Chroma client:

import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

Then create/get a collection:

collection = client.get_or_create_collection(
    name="team_knowledge"
)

A Chroma collection is where embeddings, documents and metadata are stored and indexed for retrieval. 

Now add the chunks:

collection.add(
    ids=[
        f"sample-doc-{i}"
        for i in range(len(chunks))
    ],
    embeddings=embeddings,
    documents=chunks,
    metadatas=[
        {
            "source": "sample.docx",
            "chunk": i
        }
        for i in range(len(chunks))
    ]
)

Now you have:

ChromaDB
│
├── ID
├── vector
├── original text
└── metadata

This is important:

Don't store only the vector.

Keep the original chunk and metadata.

Otherwise, when you retrieve the vector, you won't know what it represents.

Chroma's API explicitly supports IDs, embeddings, documents and metadata together. 


---

10. Complete ingest.py

For your first demo, put the pieces together:

from docx import Document
import ollama
import chromadb


EMBEDDING_MODEL = "embeddinggemma"
DOC_PATH = "data/sample.docx"


def extract_text(file_path):
    document = Document(file_path)

    parts = []

    for paragraph in document.paragraphs:
        text = paragraph.text.strip()

        if text:
            parts.append(text)

    for table in document.tables:
        for row in table.rows:
            cells = [cell.text.strip() for cell in row.cells]

            row_text = " | ".join(
                cell for cell in cells if cell
            )

            if row_text:
                parts.append(row_text)

    return "\n".join(parts)


def chunk_text(text, chunk_size=1000, overlap=150):
    chunks = []

    start = 0

    while start < len(text):
        end = start + chunk_size

        chunk = text[start:end].strip()

        if chunk:
            chunks.append(chunk)

        start += chunk_size - overlap

    return chunks


def main():

    print("Reading document...")

    text = extract_text(DOC_PATH)

    print(f"Extracted characters: {len(text)}")

    chunks = chunk_text(text)

    print(f"Created chunks: {len(chunks)}")

    print("Creating embeddings...")

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunks
    )

    embeddings = response["embeddings"]

    print("Creating ChromaDB...")

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_or_create_collection(
        name="team_knowledge"
    )

    print("Storing vectors...")

    collection.upsert(
        ids=[
            f"sample-doc-{i}"
            for i in range(len(chunks))
        ],
        embeddings=embeddings,
        documents=chunks,
        metadatas=[
            {
                "source": "sample.docx",
                "chunk": i
            }
            for i in range(len(chunks))
        ]
    )

    print("Done!")

    print("Total records:", collection.count())


if __name__ == "__main__":
    main()

I used upsert() rather than add() here deliberately.

If you rerun your ingestion script, you don't want duplicate records every time. Chroma's documentation specifically shows upsert as a convenient way to avoid adding the same documents repeatedly. 


---

11. Step 5 — Ask a question

Now comes the retrieval part.

First create the question embedding:

import ollama


EMBEDDING_MODEL = "embeddinggemma"

question = "What was the root cause of the issue?"

response = ollama.embed(
    model=EMBEDDING_MODEL,
    input=question
)

query_embedding = response["embeddings"][0]

Important: use the same embedding model used during ingestion. Ollama explicitly recommends this. 


---

12. Search ChromaDB

results = collection.query(
    query_embeddings=[query_embedding],
    n_results=3,
    include=[
        "documents",
        "metadatas",
        "distances"
    ]
)

Chroma's query operation performs nearest-neighbor search and can return documents, metadata and distances. 

Now inspect:

for i, document in enumerate(results["documents"][0]):

    print("\n--------------------")

    print("Rank:", i + 1)

    print(
        "Distance:",
        results["distances"][0][i]
    )

    print(
        "Metadata:",
        results["metadatas"][0][i]
    )

    print("Text:")
    print(document)

At this point you have implemented retrieval without generation.

That's an important debugging milestone.


---

13. Step 6 — Give the retrieved context to Ollama

Now we finally use the generative LLM.

LLM_MODEL = "gemma3"

Build the context:

documents = results["documents"][0]

context = "\n\n---\n\n".join(documents)

Then:

prompt = f"""
You are a team knowledge assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information was not found.

Question:
{question}

Context:
{context}
"""

Call Ollama:

response = ollama.chat(
    model=LLM_MODEL,
    messages=[
        {
            "role": "user",
            "content": prompt
        }
    ]
)

answer = response["message"]["content"]

print(answer)

Now you've completed the basic RAG pipeline.


---

14. Complete ask.py

import ollama
import chromadb


EMBEDDING_MODEL = "embeddinggemma"
LLM_MODEL = "gemma3"


def main():

    question = input("Question: ")

    # -----------------------------
    # 1. Connect to Chroma
    # -----------------------------

    client = chromadb.PersistentClient(
        path="./chroma_db"
    )

    collection = client.get_collection(
        name="team_knowledge"
    )

    # -----------------------------
    # 2. Embed question
    # -----------------------------

    embedding_response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=question
    )

    query_embedding = (
        embedding_response["embeddings"][0]
    )

    # -----------------------------
    # 3. Retrieve relevant chunks
    # -----------------------------

    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=3,
        include=[
            "documents",
            "metadatas",
            "distances"
        ]
    )

    documents = results["documents"][0]
    metadatas = results["metadatas"][0]
    distances = results["distances"][0]

    # -----------------------------
    # 4. Build context
    # -----------------------------

    context_parts = []

    for document, metadata, distance in zip(
        documents,
        metadatas,
        distances
    ):

        context_parts.append(
            f"""
Source: {metadata['source']}
Chunk: {metadata['chunk']}
Distance: {distance}

Content:
{document}
"""
        )

    context = "\n\n---\n\n".join(
        context_parts
    )

    # -----------------------------
    # 5. Ask LLM
    # -----------------------------

    prompt = f"""
You are a team knowledge assistant.

Answer the question using ONLY the
provided context.

If the context does not contain enough
information, say:

"I could not find sufficient evidence
in the indexed documents."

Do not invent facts.

Question:
{question}

Context:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    answer = response["message"]["content"]

    # -----------------------------
    # 6. Output
    # -----------------------------

    print("\n================ ANSWER ================\n")

    print(answer)

    print("\n================ SOURCES ================\n")

    for metadata in metadatas:
        print(
            f"- {metadata['source']} "
            f"(chunk {metadata['chunk']})"
        )


if __name__ == "__main__":
    main()


---

15. Run it

First:

python ingest.py

You should see something like:

Reading document...
Extracted characters: 18234
Created chunks: 21
Creating embeddings...
Creating ChromaDB...
Storing vectors...
Done!
Total records: 21

Then:

python ask.py

Ask:

What was the root cause of the payment failure?

You should get:

================ ANSWER ================

The root cause was ...

================ SOURCES ================

- sample.docx (chunk 7)
- sample.docx (chunk 8)

🎉 That's your first working RAG system.


---

16. How to debug it

This is actually more important than memorizing the code.

When the answer is wrong, don't immediately blame the LLM.

Debug the pipeline from left to right.

DOCX
 ↓
Extraction
 ↓
Chunks
 ↓
Embeddings
 ↓
Vector DB
 ↓
Retrieval
 ↓
Prompt
 ↓
LLM
 ↓
Answer

Find the first stage where things went wrong.


---

Problem 1: Document extraction is wrong

Run:

print(text)

Look at the extracted text.

If your Word document says:

Root Cause:
Connection pool exhausted

but Python doesn't extract it, RAG cannot fix that.

Fix ingestion first.


---

Problem 2: Chunking is bad

Print:

for i, chunk in enumerate(chunks):
    print(f"\n--- {i} ---")
    print(chunk)

Look for:

sentence starts in chunk 1
sentence ends in chunk 2

or:

heading separated from content

This tells you your chunking strategy needs improvement.


---

Problem 3: Retrieval is wrong

This is one of the most important debugging techniques.

Temporarily don't call the LLM.

Just print:

Question
 ↓
Top 5 retrieved chunks

Ask:

> "Are the correct chunks actually being retrieved?"



If the answer is no, the problem is retrieval.

If the answer is yes, but the final answer is wrong, the problem is probably prompting/context/model behavior.

This distinction will save you a LOT of debugging time.


---

Problem 4: Retrieval is good but answer is bad

Print the exact prompt:

print(prompt)

You should see:

Question:
...

Context:
[relevant chunk]
[relevant chunk]
[relevant chunk]

Then inspect whether the model has enough evidence.

Your prompt should explicitly say:

Use ONLY the supplied context.
Do not invent information.
If evidence is insufficient, say so.

This doesn't guarantee perfect behavior, but it reduces the temptation for the model to fill gaps.


---

Problem 5: Ollama connection error

Typical issue:

Connection refused

Check that Ollama is running.

Then test it independently before debugging Python.

For example, make sure your model can respond through Ollama itself.

If the Ollama CLI works but Python doesn't, then investigate the Python environment/package.


---

Problem 6: Model not found

If you see something like:

model not found

check:

ollama list

Make sure the name in Python exactly matches the installed model.

For example:

LLM_MODEL = "gemma3"

must correspond to what your local Ollama installation actually exposes.


---

Problem 7: Embedding dimension mismatch

This can happen if you created your database with one embedding model and later change models.

For example:

Old model → 768 dimensions

New model → 1024 dimensions

Don't mix them casually.

The embedding model used to index the documents must match the one used for queries. 

For a demo, the easiest fix is often:

delete chroma_db
↓
re-ingest everything

For a production system, you'd version the embedding model and re-index systematically.


---

17. How to optimize for minimal latency

This is where your backend-engineering brain should kick in.

Optimization 1 — Don't embed documents during every question

Wrong:

Question
 ↓
Read Word document
 ↓
Chunk
 ↓
Embed everything
 ↓
Search
 ↓
LLM

That would be terrible.

Instead:

ONE TIME
                    ↓
Document → chunk → embed → store


                 EVERY QUERY
                    ↓
Question → embed → search → LLM

This is one of the biggest architectural optimizations.


---

Optimization 2 — Batch embeddings

Instead of:

for chunk in chunks:
    ollama.embed(
        model=EMBEDDING_MODEL,
        input=chunk
    )

use:

ollama.embed(
    model=EMBEDDING_MODEL,
    input=chunks
)

Ollama explicitly supports batched input for embeddings. 

That reduces request overhead.

For a very large corpus, you may still batch in manageable groups rather than sending an enormous list at once.


---

18. Don't retrieve 20 chunks unnecessarily

Start with:

n_results=3

or:

n_results=5

Why?

Because giving the LLM 20 irrelevant chunks:

more tokens
+
more processing
+
more latency
+
more distraction

doesn't automatically produce a better answer.

Your goal is:

> small amount of highly relevant context.




---

19. But don't optimize accuracy too early

This is important.

You asked for:

> most accurate answer with minimal time



Those two goals can conflict.

For example:

Top 1 chunk
→ very fast
→ might miss important evidence

versus:

Top 10 chunks
→ more context
→ slower
→ may contain noise

So don't optimize blindly.

First establish:

Does it retrieve the correct information?

Then optimize latency.


---

20. The first accuracy experiment

Create 10 questions where you already know the answer.

Example:

Q1 → known answer is Issue 103
Q2 → known answer is DR 42
Q3 → known answer is section X
...

For every question record:

Question
Expected source
Retrieved source
Correct? 
Answer correct?
Latency

Now you're actually evaluating your RAG system.

That's much better than saying:

> "It feels accurate."




---

21. Don't immediately jump to advanced RAG

Your first version should be:

DOCX
 ↓
chunk
 ↓
Ollama embedding
 ↓
Chroma
 ↓
similarity search
 ↓
Ollama LLM

Then, if retrieval isn't good enough, improve it.

Potential progression:

V1
Basic vector search

        ↓

V2
Better chunking

        ↓

V3
Metadata filtering

        ↓

V4
Keyword + vector search

        ↓

V5
Reranking

        ↓

V6
Agentic/multi-step retrieval

Don't build V6 before knowing whether V1 works.


---

22. One optimization I'd eventually make for your actual project

Your eventual documents aren't just random text.

They're probably things like:

DR
Issue
Incident
Resolution
Architecture
Component
Date
Author
Severity

So store metadata:

metadata = {
    "source": "DR-42.docx",
    "type": "design_record",
    "component": "payment-service",
    "date": "2026-02-10"
}

Then you can later query:

> "Find payment-service issues."



and combine:

semantic similarity
+
metadata filtering

Chroma supports metadata filtering alongside similarity queries. 

That's going to be much more useful for your eventual organizational knowledge system than simply throwing every document into a vector database.


---

23. The most important mental model

Don't think:

> "I put a Word document into Ollama."



That's not what we're doing.

Think:

INGESTION
                     │
DOCX ──→ text ──→ chunks ──→ embeddings ──→ Vector DB
                                                │
                                                │
                  QUERY                         │
                     │                          │
Question ──→ embedding ──→ similarity search ──┘
                              │
                              ↓
                         top chunks
                              │
                              ↓
                             LLM
                              │
                              ↓
                    grounded answer

Ollama has two roles here:

Ollama
 ├── embedding model
 │      ↓
 │   vectors
 │
 └── generative model
        ↓
      answer

Chroma has one main role:

store + retrieve relevant vectors/documents

Python is orchestrating everything.


---

24. And eventually your Java application

Once you understand this Python demo, we don't need to keep the production project in Python.

You can recreate the architecture as:

Minimal UI
                        │
                        ▼
                  Spring Boot
                        │
             ┌──────────┴──────────┐
             │                     │
        Retrieval Service      LLM Service
             │                     │
             ▼                     ▼
         Vector DB               Ollama
             ▲
             │
       Ingestion Service
             ▲
             │
        Word/PDF/etc.

The Python demo is simply the fastest way for you to understand the AI mechanics before wrapping them inside your Java application.


---

Your demo's definition of "done"

Don't move on until you can do this:

sample.docx
    ↓
python
    ↓
extract text
    ↓
chunk
    ↓
Ollama embedding
    ↓
ChromaDB
    ↓
ask question
    ↓
retrieve top 3 chunks
    ↓
Ollama
    ↓
answer + source

And when something fails, you can identify whether the failure is in:

extraction → chunking → embedding → retrieval → prompt → generation.

Once you can do that, you've gone from "I watched a RAG video" to "I understand how a RAG application actually works."

For your next step after this demo, I'd make the single-document version deliberately accurate first, then add a second document and test whether retrieval can distinguish between them. Only after that should we think about your 100s of team documents.
