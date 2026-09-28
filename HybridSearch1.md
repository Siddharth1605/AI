Yep. Since you already have DOCX → chunks → Ollama embeddings → ChromaDB → vector retrieval, don't restart anything.

Your next goal is very specific:

> Take the same document and make retrieval better by combining vector search + BM25, then measure whether the hybrid system actually improves your serviceA scale → serviceA scalability case.



Your next roadmap

CURRENT
DOCX
 ↓
paragraph chunks
 ↓
Ollama embeddings
 ↓
ChromaDB
 ↓
Vector search
 ↓
Top K

You are going to reach:

DOCX
 ↓
paragraph chunks
 ├───────────────────┐
 ↓                   ↓
Ollama embedding    BM25 index
 ↓                   ↓
ChromaDB            keyword search
 ↓                   ↓
Top 10               Top 10
 └─────────┬─────────┘
           ↓
        RRF fusion
           ↓
      Top 3-5 chunks
           ↓
        Ollama LLM
           ↓
    Answer + sources


---

Step 1 — Freeze your current vector implementation

Don't modify it yet.

Make sure you can currently do:

Question
   ↓
embedding
   ↓
Chroma
   ↓
top 10 results

Change your vector retrieval temporarily to:

n_results=10

not 3.

Why?

Because your example is:

serviceA scale

1. initialization
2. configuration
3. services
4. scalability  ← relevant

If you retrieve only 3, the correct result never reaches the hybrid stage.

So for now:

> Retrieve more candidates, then rank them better.




---

Step 2 — Create a small evaluation set

Before implementing BM25, create maybe 10–20 questions for your document.

For example:

Q1: serviceA scale
Q2: why serviceA failed
Q3: serviceA configuration
Q4: serviceB timeout
Q5: how was issue X fixed?
Q6: what caused error Y?

For every question, manually identify the expected chunk/source.

Make a simple table:

Question	Expected chunk

serviceA scale	chunk 17
serviceA configuration	chunk 4
serviceB timeout	chunk 22


This is very important.

Otherwise you'll keep changing chunk size, BM25 parameters, top_k, etc. based on vibes.

Your evaluation dataset becomes your little test suite.


---

Step 3 — Understand what BM25 is doing

Don't code immediately.

Your existing vector search answers:

> "Which chunks are semantically similar to serviceA scale?"



BM25 answers:

> "Which chunks are lexically relevant to the words in serviceA scale?"



Conceptually:

serviceA scale
                         │
             ┌───────────┴───────────┐
             ↓                       ↓
       Vector search              BM25
             ↓                       ↓
       "similar meaning"       "matching terms"

That's why we're adding it.


---

Step 4 — Install BM25

Use:

pip install rank-bm25

Then:

from rank_bm25 import BM25Okapi

No Ollama is involved here.

That's an important distinction:

Ollama → embeddings / LLM

Chroma → vector retrieval

BM25 → keyword retrieval


---

Step 5 — Build your BM25 index

You already have your chunks.

For example:

chunks = [
    "ServiceA initialization process...",
    "ServiceA configuration...",
    "ServiceA services...",
    "ServiceA scalability considerations..."
]

Tokenize them:

tokenized_chunks = [
    chunk.lower().split()
    for chunk in chunks
]

Create the BM25 index:

bm25 = BM25Okapi(tokenized_chunks)

That's it.

You've created your lexical search index.


---

Step 6 — Test BM25 independently

Do not combine it with vector search yet.

Ask:

query = "serviceA scale"

query_tokens = query.lower().split()

scores = bm25.get_scores(query_tokens)

Then rank:

import numpy as np

top_indices = np.argsort(scores)[::-1][:10]

for rank, index in enumerate(top_indices, start=1):
    print(
        rank,
        scores[index],
        chunks[index]
    )

You want output like:

BM25 RESULTS

1  3.81  ServiceA scalability considerations...
2  2.94  ServiceA initialization...
3  2.73  ServiceA configuration...
...

Your actual numbers will obviously differ.

Your first experiment

Run your 10 questions through BM25.

See:

> Does BM25 retrieve the correct chunks?



If yes → great.

If no → debug BM25 before touching hybrid search.


---

Step 7 — Save the chunk IDs properly

This becomes important now.

Don't just have:

chunks = [...]

Have something like:

documents = [
    {
        "id": "doc1-chunk-0",
        "text": "...",
        "source": "DR-42.docx",
        "chunk": 0
    },
    {
        "id": "doc1-chunk-1",
        "text": "...",
        "source": "DR-42.docx",
        "chunk": 1
    }
]

Then:

texts = [
    document["text"]
    for document in documents
]

BM25 operates on texts, but returns the index.

You can map that index back to:

chunk ID
source
metadata
text

This will make the later hybrid implementation much cleaner.


---

Step 8 — Run vector and BM25 side by side

Now take:

serviceA scale

and produce two rankings.

Vector

VECTOR

1. chunk-3  ServiceA initialization
2. chunk-8  ServiceA configuration
3. chunk-11 ServiceA services
4. chunk-17 ServiceA scalability
5. chunk-22 ServiceA deployment

BM25

BM25

1. chunk-17 ServiceA scalability
2. chunk-3  ServiceA initialization
3. chunk-8  ServiceA configuration
...

This is the moment you should pause and understand what's happening.

Vector found semantic similarity.

BM25 found lexical relevance.

Neither is necessarily perfect.


---

Step 9 — Retrieve top 10 from both

Your architecture should now be:

Question
   │
   ├───────────────┐
   ↓               ↓
Vector            BM25
   ↓               ↓
Top 10            Top 10

Why 10?

Because you want to create a candidate pool.

The final answer might only use 3 chunks.

So:

Vector → 10 candidates
BM25   → 10 candidates
             ↓
          combine
             ↓
          final 3-5

This is much better than:

Vector → top 3

because your correct chunk might already have been discarded.


---

Step 10 — Now learn RRF

This is the easiest way to combine your rankings.

Suppose:

Vector:

A rank 1
B rank 2
C rank 4

and:

BM25:

C rank 1
A rank 3
D rank 4

RRF says:

> If something ranks highly in multiple retrieval systems, give it a strong combined score.



Use:

def rrf(rankings, k=60):
    scores = {}

    for ranking in rankings:
        for rank, doc_id in enumerate(ranking, start=1):
            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

You don't need to obsess over the formula yet.

Understand the principle:

Vector says:
"chunk 17 is somewhat relevant"

BM25 says:
"chunk 17 is VERY relevant"

                 ↓

        Hybrid ranking
                 ↓

       chunk 17 moves up


---

Step 11 — Test your exact problem

Now run:

serviceA scale

You want to see something like:

Before

Vector

1. initialization
2. configuration
3. services
4. scalability

After

Hybrid

1. scalability
2. initialization
3. configuration

If that happens:

you have demonstrated that hybrid retrieval solved an actual failure in your system.

That's much more meaningful than saying "I implemented BM25."


---

Step 12 — Measure it

For each question, record:

Question
Vector rank
BM25 rank
Hybrid rank

Example:

Query	Vector	BM25	Hybrid

serviceA scale	4	1	1
serviceB timeout	3	2	1
DR-142	8	1	1
configuration X	2	4	2


Now you can calculate things like Recall@3.

For example:

If the correct chunk is rank 4:

Recall@3 = miss
Recall@5 = hit

Your goal isn't necessarily:

> "Make every answer #1."



Your first retrieval goal is:

> Get the correct evidence into the candidate set reliably.



Then we worry about ranking it perfectly.


---

Step 13 — Only then connect Ollama again

Once hybrid retrieval works independently:

Question
   ↓
Vector + BM25
   ↓
RRF
   ↓
Top 3-5 chunks
   ↓
Ollama LLM

Your LLM should receive only the final relevant context.

Something like:

Question:
Why does serviceA have scalability problems?

Relevant evidence:

[Source: DR-42]
ServiceA scalability is limited because...

[Source: Issue-102]
During high load...

Answer using only the evidence above.

Then:

ollama.chat(...)


---

Step 14 — Add source citations

Don't just output:

> ServiceA has scalability issues because...



Output:

ServiceA has scalability limitations because
the worker pool is statically configured.

Sources:
- DR-42.docx — chunk 17
- Issue-102.docx — chunk 4

That is especially important for Lost Knowledge.

You're building an engineering knowledge assistant, not a generic chatbot.


---

Step 15 — Then optimize chunking

Only after hybrid search works, experiment with:

500 chars
800 chars
1000 chars
1500 chars

and:

10% overlap
15%
20%

But since you've already moved toward paragraph-based chunking, I'd prioritize structure-aware chunks over arbitrary character counts.

For example:

## ServiceA Scalability

paragraph 1
paragraph 2
paragraph 3

should preferably stay together.

Store metadata:

{
    "source": "DR-42.docx",
    "section": "ServiceA Scalability",
    "chunk_id": 17
}

This will help later with filtering and citations.


---

Step 16 — Then add metadata filtering

Eventually your documents may have:

source
document_type
component
date
author
issue_number

Then queries can become:

"ServiceA scalability"

plus:

component = "ServiceA"
document_type = "DR"

So retrieval becomes:

semantic similarity
        +
BM25
        +
metadata

That is significantly more powerful than just throwing everything into one vector database.


---

Step 17 — Then, and only then, consider reranking

Your eventual architecture could become:

Question
                      │
             ┌────────┴────────┐
             ↓                 ↓
        Vector search       BM25
             ↓                 ↓
           Top 10            Top 10
             └────────┬────────┘
                      ↓
                     RRF
                      ↓
                   Top 10
                      ↓
                  Reranker
                      ↓
                    Top 3
                      ↓
                    LLM
                      ↓
              Answer + sources

A reranker is another model specifically used to decide:

> "Of these 10 candidates, which are actually most relevant to this exact question?"



Don't learn this yet.

It's your next-next step.


---

Your actual next 3 Saturdays

Since you're doing this on Saturdays, I'd make it very concrete.

Saturday 1 — BM25

☐ Install rank-bm25
☐ Use existing chunks
☐ Build BM25 index
☐ Search 10 questions
☐ Print BM25 rankings
☐ Compare against expected chunks

No LLM changes.


---

Saturday 2 — Hybrid

☐ Vector top-10
☐ BM25 top-10
☐ Implement RRF
☐ Produce hybrid top-5
☐ Test 10–20 questions
☐ Compare vector vs hybrid

Your key experiment:

"serviceA scale"

Vector:   rank 4
Hybrid:   rank ?


---

Saturday 3 — Quality + performance

☐ Measure Recall@3 / Recall@5
☐ Measure retrieval latency
☐ Measure embedding latency
☐ Measure LLM latency
☐ Experiment with chunk size
☐ Experiment with top-K
☐ Add source metadata
☐ Return source citations

At that point you'll have something you can actually explain in an interview:

> "I initially used semantic vector retrieval with Ollama embeddings and found that exact engineering terms weren't always ranked correctly. I added BM25 lexical retrieval and combined the candidate rankings using reciprocal rank fusion. I evaluated vector-only versus hybrid retrieval on a manually labeled question set and measured retrieval recall and latency."



That is a genuinely good AI/backend project story for a 1.2 YOE engineer.

And notice the progression: you're not trying to learn "all of AI." You're learning exactly enough information retrieval + LLM application engineering to make your own system measurably better.
