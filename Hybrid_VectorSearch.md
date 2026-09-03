
> Query: "serviceA scale"
Relevant chunk: "serviceA scalability" → rank 4
Irrelevant chunks: "serviceA initialization", "service configuration" → ranks 1–3



This is a perfect reason to learn hybrid retrieval rather than blindly increasing top_k.

I’ll structure this as a mini implementation document, starting from why BM25 exists and ending with a Python implementation you can plug into your current Ollama + Chroma demo.

Hybrid Search for Your RAG POC

1. The problem we're solving

Your current pipeline is approximately:

Question
   ↓
Ollama embedding
   ↓
Vector DB
   ↓
Top K chunks
   ↓
LLM

Vector search asks essentially:

> "Which chunks have meanings most similar to this question?"



That's powerful, but it doesn't necessarily understand that certain exact words are especially important.

Your query:

serviceA scale

contains two pieces of information:

serviceA   → exact entity/component
scale      → important keyword

But your vector search can consider:

serviceA initialization
serviceA configuration
serviceA scalability

all semantically related.

So you get:

1. serviceA initialization
2. serviceA configuration
3. serviceA services
4. serviceA scalability     ← actual answer

Increasing top_k from 3 → 10 may make the correct chunk appear, but it doesn't necessarily make it rank higher.

That's the distinction.


---

2. What is keyword search?

Before BM25, forget AI for a moment.

Imagine your document contains:

ServiceA initialization process
ServiceA configuration
ServiceA scalability considerations
ServiceB deployment

Query:

serviceA scale

A simple keyword search looks for words appearing in the documents.

It might notice:

Document                         Matches

ServiceA initialization         serviceA
ServiceA configuration          serviceA
ServiceA scalability             serviceA
                                  ↑
                              scale-related

The problem is that ordinary keyword matching doesn't understand language very well.

scale and scalability are different strings.

This is where BM25 becomes useful.


---

3. What is BM25?

BM25 is a classic lexical information-retrieval ranking algorithm.

Don't let the name scare you.

The basic idea is:

> How relevant is this document to these exact words?



BM25 considers things such as:

1. Does the query term appear?

If the document contains:

serviceA

that's good.

2. How often does it appear?

A term appearing several times can be useful, although BM25 deliberately prevents repeated words from increasing the score forever.

3. Is the word rare across the corpus?

This is important.

Suppose:

serviceA

appears in 900 out of 1,000 chunks.

It isn't very useful for distinguishing documents.

But:

scalability

appears in only 20 chunks.

That's much more informative.

BM25 therefore gives more importance to terms that distinguish documents.

4. Does document length matter?

BM25 normalizes for document length so that long documents don't automatically win just because they contain more words.


---

4. BM25 vs vector search

This is the important mental model.

Vector search

Understands semantic similarity.

Query:
"serviceA scale"

May find:

"serviceA scalability"
"serviceA capacity planning"
"serviceA performance under load"

Even if the exact words aren't identical.

BM25

Understands lexical relevance.

Query:
"serviceA scale"

Looks strongly at:

serviceA
scale

and related exact lexical occurrences.

So:

Search
                   │
          ┌────────┴────────┐
          ↓                 ↓
     Vector search       BM25
     "meaning"           "words"
          │                 │
          ↓                 ↓
      semantic           lexical
      candidates         candidates
          │                 │
          └────────┬────────┘
                   ↓
             Hybrid ranking
                   ↓
              best chunks

That's hybrid search.


---

5. Why hybrid is better for your knowledge system

Your documents will probably contain lots of things like:

ServiceA
Kafka
DR-142
Issue-348
payment-service
connectionPool
HTTP 502
timeout
config-X

These are entities and identifiers.

Vector search can be surprisingly bad at exact identifiers.

For example:

"DR-142"

is not really a semantic concept.

You often want:

> Find the chunks containing exactly DR-142.



That's where lexical retrieval shines.

Meanwhile:

> "Why did we experience repeated failures after deployment?"



might not contain the exact wording used in the document.

That's where semantic retrieval shines.

Therefore:

Vector search
    +
BM25

is a very sensible architecture for your project.


---

6. Your new architecture

Your old architecture:

Question
   ↓
Embedding
   ↓
Chroma
   ↓
Top K
   ↓
LLM

becomes:

Question
                            │
                   ┌────────┴────────┐
                   ↓                 ↓
             Vector search        BM25
                   │                 │
                   ↓                 ↓
              semantic           keyword
              candidates         candidates
                   │                 │
                   └────────┬────────┘
                            ↓
                     Combine rankings
                            ↓
                       Top results
                            ↓
                           LLM
                            ↓
                    Grounded answer

This is the first version I'd build.


---

7. Don't throw away Chroma

You still need your existing vector DB.

Keep:

Chroma
 ↓
vector similarity

Add:

BM25
 ↓
lexical search

The two systems can operate independently.

For your current POC, you can implement BM25 locally in Python.

A simple library is:

pip install rank-bm25

Then:

from rank_bm25 import BM25Okapi


---

8. Build the BM25 index

Suppose Chroma contains:

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

Then:

bm25 = BM25Okapi(tokenized_chunks)

Now you have a BM25 index.


---

9. Search using BM25

User asks:

query = "serviceA scale"

Tokenize:

query_tokens = query.lower().split()

Search:

scores = bm25.get_scores(query_tokens)

You might get something conceptually like:

Chunk                              BM25 score

ServiceA initialization             0.8
ServiceA configuration              0.7
ServiceA services                   0.6
ServiceA scalability                1.9

The exact numbers don't matter.

What matters is:

scalability → higher lexical relevance


---

10. Retrieve the top BM25 results

import numpy as np

top_indices = np.argsort(scores)[::-1][:5]

for index in top_indices:
    print(
        index,
        scores[index],
        chunks[index]
    )

[::-1] means we're sorting descending.

So now you have:

BM25
 ↓
Top 5 keyword matches


---

11. But now we have TWO rankings

Suppose vector search gives:

Vector ranking

1. initialization
2. configuration
3. services
4. scalability
5. deployment

And BM25 gives:

BM25 ranking

1. scalability
2. initialization
3. configuration
4. services
5. deployment

How do we combine them?

This is the next important concept.


---

12. Don't simply add raw scores

You might initially think:

final_score = vector_score + bm25_score

Don't do that directly.

Why?

Because their scores aren't necessarily comparable.

For example:

Vector distance:
0.31

BM25:
4.82

You can't meaningfully say:

0.31 + 4.82

The scales and meanings are different.

This is why hybrid search commonly uses rank-based fusion or score normalization.

For your first implementation, I'd use Reciprocal Rank Fusion (RRF).


---

13. What is RRF?

Don't worry about the mathematics.

The idea is:

> A result that appears near the top in multiple ranking systems gets a strong combined ranking.



Imagine:

Vector:

scalability       rank 4
initialization    rank 1
configuration    rank 2

BM25:

scalability       rank 1
initialization    rank 2
configuration    rank 3

scalability gets:

rank 4 from vector
rank 1 from BM25

So both systems are saying:

> "This document is relevant."



That is valuable.


---

14. RRF formula

The basic formula is:

RRF(d) = Σ 1 / (k + rank(d))

You don't need to memorize it.

Usually:

k = 60

is used as a reasonable starting value.

The important part is:

rank 1 → larger contribution
rank 2 → slightly smaller
rank 3 → smaller
...

So we reward things appearing near the top.


---

15. Implement RRF

def reciprocal_rank_fusion(
    rankings,
    k=60
):
    scores = {}

    for ranking in rankings:

        for rank, doc_id in enumerate(
            ranking,
            start=1
        ):

            scores[doc_id] = (
                scores.get(doc_id, 0)
                + 1 / (k + rank)
            )

    return sorted(
        scores.items(),
        key=lambda x: x[1],
        reverse=True
    )

Now:

vector_ranking = [
    10,
    5,
    7,
    2,
    8
]

bm25_ranking = [
    2,
    10,
    7,
    5,
    3
]

fused = reciprocal_rank_fusion(
    [
        vector_ranking,
        bm25_ranking
    ]
)

print(fused)

Notice:

Document 2
Vector rank = 4
BM25 rank = 1

It receives strong combined evidence.


---

16. But there's an important issue with your current Chroma setup

Your Chroma query might return only:

n_results=3

Don't do that anymore.

If the correct result is rank 4 in vector search, you need to give your hybrid system enough candidates.

Use something like:

n_results=10

for each retrieval method.

Then:

Vector → top 10
BM25   → top 10
       ↓
    combine
       ↓
   final top 3

This is called candidate retrieval followed by fusion.


---

17. Complete hybrid retrieval architecture

Your query path now becomes:

User question
                        │
               "serviceA scale"
                        │
              ┌─────────┴─────────┐
              │                   │
              ▼                   ▼
       Query embedding       Query tokens
              │                   │
              ▼                   ▼
        Chroma search           BM25
              │                   │
           Top 10               Top 10
              │                   │
              └─────────┬─────────┘
                        ↓
                  RRF fusion
                        ↓
                  Top 3 chunks
                        ↓
                 Context creation
                        ↓
                     Ollama
                        ↓
                     Answer

This is the system I'd build next.


---

18. One important improvement: tokenization

Our:

query.lower().split()

is deliberately primitive.

For:

"serviceA scale"

it produces:

["servicea", "scale"]

Fine.

But real engineering documents contain:

serviceA
ServiceA
service-A
service_A
HTTP-502
connection_pool

You eventually want better tokenization.

For the first POC, don't over-engineer this.

Later you can normalize:

ServiceA → servicea
service-A → servicea

only if doing so makes sense for your domain.

Be careful with identifiers: sometimes punctuation is meaningful.


---

19. Your chunking experiment matters here

You said you've already increased chunk length and moved toward paragraph-based chunking.

Good.

For your organizational documents, I'd prefer structure-aware chunking over arbitrary character splitting.

For example:

Heading
    ↓
paragraph
paragraph
paragraph

Next heading
    ↓
paragraph
paragraph

Keep a chunk like:

{
    "text": "...",
    "source": "DR-42.docx",
    "section": "ServiceA Scalability",
    "chunk_id": 17
}

Then your retrieval system can use:

semantic similarity
+
BM25
+
metadata

That's becoming a serious retrieval pipeline.


---

20. Don't make the LLM solve retrieval problems

This is a key architectural principle.

Bad approach:

Retrieve 20 vaguely relevant chunks
             ↓
             LLM
             ↓
"Please figure out which one is relevant"

You're making the expensive generative model do retrieval work.

Better:

Question
 ↓
Vector + BM25
 ↓
good candidate ranking
 ↓
3–5 highly relevant chunks
 ↓
LLM

The retrieval layer should do as much filtering as possible before the LLM.


---

21. But what if hybrid still ranks scalability at #4?

That's where your debugging becomes interesting.

You should inspect:

Query

then print:

VECTOR RESULTS

Rank 1
score:
source:
text:

Rank 2
...

BM25 RESULTS

Rank 1
score:
source:
text:

Rank 2
...

RRF RESULTS

Rank 1
...

You want to understand why the result is ranked where it is.

Don't just keep changing parameters until it "looks good."


---

22. A proper evaluation dataset

This is something I strongly recommend you start doing now.

Create:

evaluation.json

with perhaps 20 questions:

[
  {
    "question": "serviceA scale",
    "expected_source": "DR-42",
    "expected_chunk": 17
  },
  {
    "question": "Why was serviceB changed?",
    "expected_source": "DR-31",
    "expected_chunk": 8
  }
]

Then measure:

Question
Vector rank
BM25 rank
Hybrid rank
Correct?
Latency

For example:

Query                 Vector   BM25   Hybrid

serviceA scale           4       1       1
serviceB timeout         2       5       2
connection pooling      7       2       2
DR-142                   8       1       1

Now you're doing retrieval engineering, not just building a demo.


---

23. Measure Recall@K

This is the first retrieval metric I'd learn.

Suppose the correct chunk is:

rank 4

Then:

Recall@3 = 0
Recall@5 = 1

Meaning:

> Did the correct document appear within my top K results?



For your current problem:

Vector top 3
→ fails

Vector top 5
→ succeeds

But after hybrid:

Hybrid top 3
→ succeeds

That's an actual measurable improvement.


---

24. Latency measurement

You also specifically care about speed.

Measure each stage:

import time

start = time.perf_counter()

# embedding
...

embedding_time = (
    time.perf_counter() - start
)

Then:

Embedding:       80 ms
Vector search:    5 ms
BM25 search:      2 ms
RRF:              <1 ms
LLM generation: 900 ms

Total:          ~987 ms

Now you know where your latency actually goes.

Don't optimize something that takes 2 ms when your LLM takes 900 ms.


---

25. Your eventual optimized pipeline

After you've proven basic hybrid search, I'd aim toward:

Question
                            │
                            ▼
                     Query normalization
                            │
                ┌───────────┴───────────┐
                ▼                       ▼
          Vector retrieval         BM25 retrieval
                │                       │
              Top 10                  Top 10
                │                       │
                └───────────┬───────────┘
                            ▼
                       RRF fusion
                            │
                         Top 5–10
                            │
                            ▼
                        Reranker
                            │
                          Top 3
                            │
                            ▼
                     Context assembly
                            │
                            ▼
                          Ollama
                            │
                            ▼
                    Answer + citations

But don't implement the reranker yet.

Your learning progression should be:

✅ Vector search
        ↓
✅ Better chunking
        ↓
👉 BM25
        ↓
👉 Hybrid/RRF
        ↓
👉 Evaluation
        ↓
👉 Metadata filtering
        ↓
👉 Reranking
        ↓
👉 Agentic retrieval

That is a much more sensible progression than jumping straight to agents.


---

26. Where Ollama fits

One thing that may be confusing:

BM25 doesn't need Ollama.

Your pipeline is:

Ollama
                /      \
               /        \
        embeddings      LLM
             ↓           ↓
         Chroma        answer

while:

BM25
 ↓
Python/local index

So:

Question
                     │
          ┌──────────┴──────────┐
          ↓                     ↓
      Ollama embed           BM25
          ↓                     ↓
       Chroma                ranking
          │                     │
          └──────────┬──────────┘
                     ↓
                   RRF
                     ↓
              relevant chunks
                     ↓
                  Ollama
                     ↓
                  answer

That is a very nice architecture for your local POC.


---

Your next Saturday should be very specific

Don't learn "hybrid search" broadly for five hours.

Do this:

Part 1 — 30 min

Understand:

lexical search → TF-IDF → BM25

You don't need to implement TF-IDF yourself.

Part 2 — 45 min

Install:

pip install rank-bm25

and make BM25 work on your existing chunks.

Part 3 — 60 min

Run the same 10 questions through:

Vector only
BM25 only

Print the rankings.

Part 4 — 60 min

Implement RRF.

Compare:

Vector
BM25
Hybrid

Part 5 — 30 min

Create your evaluation table:

Query | Vector Rank | BM25 Rank | Hybrid Rank | Latency

Part 6 — remaining time

Tune:

chunk size

overlap

top-K candidates

final K

BM25 tokenization


Don't touch agents yet.

You're currently learning something much more valuable: how to make retrieval actually work.

And your serviceA scale → serviceA scalability is #4 example is exactly the kind of concrete failure that should drive the next engineering iteration. That's how I'd build this project rather than blindly following an RAG tutorial.
