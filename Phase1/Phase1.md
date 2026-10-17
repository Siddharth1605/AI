Let's start over and understand it slowly. Forget the technical terms for a moment. We'll use one example throughout, then connect every step to your actual Python code.

# 1. First, understand what you're building

Imagine your company has **20 SRS documents**. One document explains a database service, another explains image synchronization, and another explains error handling.

An engineer asks:

> What is the default connection pool size for Service A?

Today, someone might have to open several Word documents and search manually.

You want your application to find the correct information and answer the question.

There is one problem: **Gemma 3 doesn't automatically know what's written in your private SRS documents.** You need to give it the relevant information.

That's the job of your RAG application.

## The entire idea in one diagram



Think of your application as a technical librarian.

- **Your documents** are the books.
- **Ingestion** is the process of organizing and indexing those books.
- **Retrieval** is the librarian finding the relevant pages.
- **The LLM** reads those pages and explains the answer.
- **The final response** tells the engineer what the documents say.

Your application has two major flows:

1. **Preparation:** Organize the documents before anyone asks a question.
2. **Question answering:** Find relevant information whenever someone asks a question.

Let's understand them separately.

---

# 2. Flow One: Preparing your documents

This happens when you run `ingest.py`.

Imagine you have these three documents:

```text
documents/
├── Database_Service_SRS.docx
├── Image_Sync_SRS.docx
└── MQTT_Service_SRS.docx
```

Each document contains many paragraphs, headings and tables.

You cannot simply ask a vector database to understand a Word file directly. First, your code must extract and organize its contents.

## Step 1: Read the Word documents

Your Python code opens each DOCX file and extracts its paragraphs and table rows.

Suppose `Database_Service_SRS.docx` contains:

```text
Section 1: Overview

Service A connects to PostgreSQL.

Section 2: Configuration

The default connection pool size is 20.

Section 3: Error Handling

The service retries failed connections.
```

At this point, the program has extracted text from the Word document.

It has not generated an answer. It has not searched for anything. It is simply reading the files.

**In your code:** `extract_docx_blocks()` handles the text extraction.

## Step 2: Divide the document into meaningful sections

Now the code identifies headings and groups the content under them.

Imagine the document is organized like this:



The result is conceptually:

```text
Database_Service_SRS.docx
│
├── Section 1: Overview
│   └── Service A connects to PostgreSQL.
│
├── Section 2: Configuration
│   └── The default connection pool size is 20.
│
└── Section 3: Error Handling
    └── The service retries failed connections.
```

Why do this?

Suppose someone asks about the connection pool size. You want to find the Configuration section, not an unrelated paragraph about connection failures.

Keeping sections separate makes the information easier to retrieve.

**In your code:** `build_sections()` performs this grouping.

## Step 3: Create chunks

A *chunk* is a smaller piece of text that can be searched independently.

Why not use the entire document as one chunk?

Because a document can contain hundreds of different requirements. If you store everything as one giant piece, finding the exact paragraph becomes harder.

Your code splits large sections into smaller chunks when necessary. It also preserves the section heading in each chunk.

For example:

```text
DOCUMENT
Database_Service_SRS.docx
       │
       ▼
SECTION: Configuration
       │
       ▼
CHUNK 1
"The default connection pool size is 20."
```

A different section might produce another chunk:

```text
CHUNK 2
"Failed database connections are retried."
```

These chunks are separate searchable records.

An important detail: your code does **not** necessarily make one chunk for every sentence. It keeps a section together when it fits within the configured limit and splits it when it becomes too large.

**In your code:** `split_section()` creates the chunks.

## Step 4: Convert every chunk into numbers

Computers can compare numerical vectors to find text with similar meaning. These vectors are called **embeddings**.

Your code sends each chunk to the `nomic-embed-text` model through Ollama.

For example:

```text
CHUNK 1
"The default connection pool size is 20."
                │
                ▼
        nomic-embed-text
                │
                ▼
       Numerical vector
       [0.12, -0.44, 0.81, ...]
```

The numbers above are illustrative, not actual output.

The embedding represents aspects of the text's meaning. Texts with related meanings may have similar vectors.

For instance, these two sentences use different words but are related:

- "The default connection pool size is 20."
- "What is the maximum number of database connections?"

Their embeddings may be similar because they concern database connections.

**Important:** The embedding model doesn't answer questions. It only converts text into numerical representations that help the application search.

**In your code:** `create_embeddings_in_batches()` generates these vectors.

## Step 5: Store the chunks and vectors in ChromaDB

Now we need somewhere to keep the information.

Your project uses **ChromaDB**, a database that stores your text chunks, their metadata and their embeddings.

Think of each stored record as a card in a library catalogue.

For example:

| Information stored | Example |
|---|---|
| Chunk ID | `a81f...` |
| Document | `Database_Service_SRS.docx` |
| Section | `Configuration` |
| Text | The default connection pool size is 20. |
| Embedding | `[0.12, -0.44, 0.81, ...]` |

The embedding lets ChromaDB search by meaning. The text gives your application the actual evidence. The metadata tells it where that evidence came from.

Your code uses a stable identifier for each chunk, derived from the document, section, child index and content.

### What happens after ingestion?

Your documents have been converted into searchable records.

```text
20 SRS documents
       │
       ▼
Extract paragraphs and tables
       │
       ▼
Identify sections
       │
       ▼
Create chunks
       │
       ▼
Generate embeddings
       │
       ▼
Store chunks + embeddings + metadata
       │
       ▼
    ChromaDB
       │
       ▼
READY TO ANSWER QUESTIONS
```

That's the first half of your application.

You run `ingest.py` to prepare the knowledge base. You don't need to repeat this entire process every time an engineer asks a question.

One detail about your current code: it rebuilds the collection and generates embeddings for all chunks on every ingestion run. We can optimize that later.

---

# 3. Flow Two: Answering a question

Now let's leave ingestion behind.

Your documents are already stored in ChromaDB. An engineer opens `ask.py` and asks:

> What is the default connection pool size for Service A?

Let's follow this exact question through your code.

## Step 1: Load the searchable knowledge

When `ask.py` starts, it creates a `HybridRetriever`.

The retriever loads the existing chunks from ChromaDB and builds another search structure called a BM25 index.

Why another search structure?

Because your application uses **two different ways to find relevant text**:

1. Search by meaning.
2. Search by exact words.

We'll see why both are useful in a moment.

This loading and indexing happens when the program starts, not from scratch for every question.

## Step 2: Understand what the question is asking

Your code calls `analyze_question()`.

This function sends the question to Gemma 3 and asks it to identify three things:

- **Entity:** What service or component are we talking about?
- **Intent:** What information do we want?
- **Keywords:** Which terms might help us find the information?

For our example, the model might produce something like this:

```text
Original question:
"What is the default connection pool size for Service A?"

                │
                ▼

       QUESTION ANALYSIS
       ─────────────────
       Entity: Service A
       Intent: Configuration
       Keywords: connection pool, default, size
```

This output is illustrative. The actual result depends on the model.

Why analyze the question first?

Because engineering documents can contain similar terms in different contexts.

For example, the word "scale" could refer to a physical weighing scale or the ability of a software service to handle more traffic. Understanding the full question helps distinguish them.

Your code combines the original question with the extracted information to construct an expanded search query.

**Remember:** Gemma 3 is not answering the user's question here. It is helping the retrieval system search more effectively.

---

# 4. The most important part: How your application finds the answer

This is the part that often makes RAG confusing.

You have hundreds of chunks in your database. Your application must decide which ones are worth showing to the LLM.

Your project uses a sequence of retrieval and ranking stages.

## Step 3: Search by meaning

The expanded question is sent to `nomic-embed-text` to create a query embedding.

```text
Question
   │
   ▼
nomic-embed-text
   │
   ▼
Question embedding
   │
   ▼
ChromaDB compares it with stored embeddings
   │
   ▼
Top 20 vector-search candidates
```

Imagine your database contains these chunks:

| Chunk | Text |
|---|---|
| A | The default connection pool size is 20. |
| B | The service supports database connection pooling. |
| C | The service retries failed database connections. |
| D | Image synchronization runs every 30 seconds. |

Vector search ranks chunks according to semantic similarity with the question.

It might find A and B useful because they discuss connection pooling. Depending on the text and embeddings, it could also return other related passages.

The important point is that **vector search finds candidates, not necessarily the final answer**.

It can recognize related meanings, but semantic similarity alone doesn't guarantee that a chunk contains the exact property or value requested.

## Step 4: Search by exact words

Your application also searches using BM25.

BM25 is a keyword-based ranking algorithm. It examines which words appear in the question and which appear in each stored chunk.

For example, the question contains:

```text
default
connection
pool
size
Service A
```

A chunk containing the exact property name or relevant technical terms may receive a strong BM25 ranking.

```text
Question
   │
   ▼
Extract search terms
   │
   ▼
BM25 index
   │
   ▼
Rank chunks by keyword relevance
   │
   ▼
Top 20 BM25 candidates
```

Why is this useful?

Suppose a document contains a configuration property named:

`db.connection.pool.size`

Vector search might find a semantically related paragraph that doesn't mention the property itself.

BM25 can help retrieve the chunk containing the exact property name.

### Vector search versus BM25

| Vector search | BM25 |
|---|---|
| Searches by semantic similarity | Searches by lexical matching |
| Can find related wording | Good for exact technical terms |
| Uses embeddings | Uses tokens and term statistics |
| Searches through ChromaDB | Uses your in-memory BM25 index |

Neither method is perfect on its own.

That's why you combine them.

## Step 5: Combine both search results using RRF

At this point, your application has two ranked lists.

Imagine they look like this:

**Vector search**

| Rank | Chunk |
|---:|---|
| 1 | B — General connection pooling |
| 2 | A — Default pool size is 20 |
| 3 | C — Connection retries |

**BM25 search**

| Rank | Chunk |
|---:|---|
| 1 | A — Default pool size is 20 |
| 2 | C — Connection retries |
| 3 | B — General connection pooling |

Notice something interesting.

Chunk A wasn't first in vector search, but it was first in BM25. Chunk B was first in vector search, but it ranked lower in BM25.

Which should your application prefer?

Instead of choosing one search method, your code combines the rankings using **Reciprocal Rank Fusion (RRF)**.

RRF gives each candidate a score based on its position in the two lists. Candidates that rank well in both searches can benefit from both contributions.

```text
Vector search results         BM25 search results
       │                             │
       ▼                             ▼
     Rank list A                   Rank list B
       │                             │
       └──────────────┬──────────────┘
                      ▼
             RECIPROCAL RANK
                 FUSION
                      │
                      ▼
              Combined ranking
                      │
                      ▼
              Top 15 candidates
```

In this example, Chunk A is a strong candidate because it ranks highly in both lists.

Your implementation retains up to 15 candidates after fusion.

**Think of RRF as combining the opinions of two search methods.** It doesn't read the documents like a human or determine the final answer. It combines their rankings mathematically.

---

# 5. Why do you need more ranking after RRF?

Here's where your project becomes more specific to engineering documentation.

Imagine the question is:

> What is the default connection pool size for Service A?

Your search results contain these passages:

| Chunk | Section title | Content |
|---|---|---|
| A | Database Configuration | Default pool size is 20. |
| B | Database Performance | Database connection pooling improves performance. |
| C | Error Handling | Failed connections are retried. |
| D | Image Synchronization | Images are synchronized periodically. |

Chunks A, B and C are all related to databases, but they aren't equally useful.

Chunk A directly contains the requested value. Chunk B provides general background. Chunk C discusses a different topic.

Your application needs to distinguish **a passage that mentions the subject from a passage that actually answers the question**.

Your code does this in two stages.

## Step 6: Metadata reranking

Your `metadata_rerank()` function examines the preliminary candidates and calculates additional scores.

It looks at things such as:

- Does the section title match the requested entity?
- Does the title match the requested intent?
- Do the filename or section ID match?
- Do the body text and keywords match?

For example:

```text
Question:
"What is the default connection pool size for Service A?"

                  │
                  ▼

       Metadata reranking
                  │
        ┌─────────┴─────────┐
        ▼                   ▼
 Section title          Chunk content
        │                   │
        ▼                   ▼
 Database             Mentions exact
 Configuration        property and value
        │                   │
        └─────────┬─────────┘
                  ▼
          Higher relevance
```

The implementation combines the metadata score with the RRF score and sorts the candidates again.

This is ordinary Python scoring logic. It doesn't require another LLM call.

The purpose is to use the structure of the SRS document to improve the ranking.

## Step 7: LLM reranking

Now your code has a preliminary ranking based on retrieval scores and metadata.

Your implementation takes up to eight candidates and sends their content and metadata to Gemma 3.

The prompt asks the model to judge which passages contain the most direct evidence for the question.

For example:

```text
QUESTION
What is the default connection pool size for Service A?

CANDIDATE A
Section: Database Configuration
Content: Default connection pool size is 20.

CANDIDATE B
Section: Database Performance
Content: Connection pooling improves performance.

CANDIDATE C
Section: Error Handling
Content: Failed database connections are retried.
```

Gemma 3 is asked to rank these candidates by relevance.

It should prefer Candidate A because it contains the exact requested information.

This is different from the earlier question-analysis call:

| Question analysis | LLM reranking |
|---|---|
| Examines the user's question | Examines the question and candidate passages |
| Produces entities, intent and keywords | Scores and ranks candidate passages |
| Helps prepare the search query | Helps choose the best evidence |
| Runs before retrieval | Runs after retrieval |

The reranker returns a ranking. It does not generate the final user-facing answer.

Your code then selects the top four chunks for the final answer.

### The complete retrieval process

Here's the whole sequence together:

```text
                     USER QUESTION
                           │
                           ▼
                    Gemma 3 analysis
                  Entities, intent, keywords
                           │
                           ▼
                    Expanded query
                           │
               ┌───────────┴───────────┐
               ▼                       ▼
         VECTOR SEARCH             BM25 SEARCH
         Search by meaning         Search by words
               │                       │
               ▼                       ▼
           Top 20                  Top 20
               │                       │
               └───────────┬───────────┘
                           ▼
                          RRF
                           │
                           ▼
                  Top 15 candidates
                           │
                           ▼
                 Metadata reranking
                           │
                           ▼
                  Top 8 candidates
                           │
                           ▼
                  Gemma 3 reranking
                           │
                           ▼
                    Final 4 chunks
```

This is the complete retrieval side of your application.

Notice that the system has not answered the user's question yet. It has only found and ranked evidence.

---

# 6. Finally, generate the answer

Now we reach the last stage.

Your `ask.py` code receives the four selected chunks from `retrieval.py`.

It formats them into a prompt containing the question and the supporting source information.

For example:

```text
You are an engineering knowledge assistant.

Answer using only the supplied sources.

Question:
What is the default connection pool size for Service A?

Sources:

[SOURCE 1]
Document: Database_Service_SRS.docx
Section: Database Configuration

The default connection pool size is 20.

[SOURCE 2]
Document: Database_Service_SRS.docx
Section: Database Performance

The service uses connection pooling to manage database
connections.

Answer:
```

Gemma 3 receives this prompt and generates the final response.

A possible answer is:

> The default connection pool size for Service A is 20. [SOURCE 1]

The answer is based on the retrieved text, and the source label identifies which supplied passage supports the claim.

Your prompt also instructs the model to say:

`Information not found in the available documents.`

when the supplied sources do not contain the answer.

This is an important design choice: you want the model to use the retrieved documents rather than inventing a configuration value from general knowledge.

However, instructions alone cannot guarantee that the model will always behave correctly. You should test whether it cites the right source and handles missing information properly.

---

# 7. Put everything together

Now that you've seen the individual steps, here is the entire system as one diagram.



And here is the same architecture mapped directly to your project:

```text
                    PHASE 1: PREPARATION
                    ====================

20 SRS DOCX files
        │
        ▼
     ingest.py
        │
        ▼
Extract paragraphs and tables
        │
        ▼
Group content into sections
        │
        ▼
Split large sections into chunks
        │
        ▼
Generate embeddings
(nomic-embed-text through Ollama)
        │
        ▼
Store chunks, metadata and vectors
        │
        ▼
      ChromaDB


                   PHASE 2: QUESTION
                   =================

Engineer asks a question
        │
        ▼
      ask.py
        │
        ▼
HybridRetriever.search()
        │
        ▼
Question analysis (Gemma 3)
        │
        ▼
Expanded search query
        │
        ├──────────────────┐
        ▼                  ▼
   Vector search       BM25 search
        │                  │
        └────────┬─────────┘
                 ▼
                RRF
                 │
                 ▼
        Metadata reranking
                 │
                 ▼
         LLM reranking
                 │
                 ▼
         Select top 4 chunks
                 │
                 ▼
       Construct answer prompt
                 │
                 ▼
        Answer generation
             (Gemma 3)
                 │
                 ▼
      Answer with source labels
```

## 8. The five things you should remember

If the entire explanation feels like too much, remember these five points:

**1. Ingestion prepares the knowledge.**  
It extracts text, creates chunks, generates embeddings and stores everything in ChromaDB.

**2. Embeddings represent text numerically.**  
They let your application search for passages with related meanings.

**3. Retrieval finds possible answers.**  
Vector search finds semantically related passages, while BM25 finds keyword matches.

**4. Reranking chooses better evidence.**  
RRF combines the search rankings. Metadata and the optional LLM reranker then help prioritize the most relevant passages.

**5. The final LLM generates the answer.**  
Gemma 3 receives the selected passages and produces a response based on them.

The distinction that matters most is this:

**Searching for relevant text and generating an answer are two different jobs.** Your RAG application performs the search first, then gives the selected evidence to the LLM.

Once you understand that separation, the rest of your code becomes much easier to follow.
