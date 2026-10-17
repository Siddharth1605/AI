import json
import re
from pathlib import Path

import chromadb
import ollama
from rank_bm25 import BM25Okapi


# ============================================================
# CONFIGURATION
# ============================================================

BASE_DIRECTORY = Path(__file__).resolve().parent
CHROMA_DIRECTORY = BASE_DIRECTORY / "chroma_db"

COLLECTION_NAME = "engineering_knowledge"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "gemma3"

# Initial candidates retrieved separately by vector and BM25.
VECTOR_CANDIDATES = 20
BM25_CANDIDATES = 20

# Number retained after Reciprocal Rank Fusion.
FUSED_CANDIDATES = 15

# Number passed to the LLM reranker.
LLM_RERANK_CANDIDATES = 8

# Final chunks returned for answer generation.
FINAL_CONTEXT_CHUNKS = 4

# RRF settings.
RRF_CONSTANT = 60
VECTOR_WEIGHT = 0.55
BM25_WEIGHT = 0.45

# Set to False if reranking is too slow during debugging.
ENABLE_LLM_QUERY_ANALYSIS = True
ENABLE_LLM_RERANKING = True

# Number of characters from each candidate sent to the reranker.
RERANK_CONTENT_LIMIT = 2200


# ============================================================
# STOPWORDS
# ============================================================

STOPWORDS = {
    "a",
    "an",
    "the",
    "is",
    "are",
    "was",
    "were",
    "be",
    "been",
    "being",
    "to",
    "of",
    "in",
    "on",
    "at",
    "for",
    "from",
    "with",
    "by",
    "and",
    "or",
    "as",
    "into",
    "about",
    "how",
    "what",
    "why",
    "when",
    "where",
    "which",
    "who",
    "does",
    "do",
    "did",
    "it",
    "this",
    "that",
    "these",
    "those",
    "can",
    "could",
    "shall",
    "should",
    "would"
}


# ============================================================
# TEXT PROCESSING
# ============================================================

def tokenize(text):
    """
    Tokenize ordinary and technical terms.

    Examples preserved as whole tokens:

        800-rtt
        mqtt-rtt
        db.connection.pool.size
        sync.client.ips
        outofmemoryerror

    Compound terms are also broken into parts.

    Example:

        db.connection.pool.size

    becomes:

        db.connection.pool.size
        db
        connection
        pool
        size
    """

    if not text:
        return []

    lowered_text = text.lower()

    raw_tokens = re.findall(
        r"[a-z0-9]+(?:[._/-][a-z0-9]+)*",
        lowered_text
    )

    tokens = []

    for token in raw_tokens:
        if token not in STOPWORDS:
            tokens.append(token)

        token_parts = re.split(
            r"[._/-]",
            token
        )

        if len(token_parts) > 1:
            for part in token_parts:
                if (
                    part
                    and part not in STOPWORDS
                ):
                    tokens.append(part)

    return tokens


def normalize_for_matching(text):
    """
    Produce normalized text for exact and partial matching.
    """

    if not text:
        return ""

    return re.sub(
        r"[^a-z0-9]+",
        " ",
        text.lower()
    ).strip()


def token_overlap(query_text, target_text):
    """
    Calculate how many important query terms occur in the
    target text.

    Result is from 0.0 to 1.0.
    """

    query_tokens = set(tokenize(query_text))
    target_tokens = set(tokenize(target_text))

    if not query_tokens:
        return 0.0

    matching_tokens = (
        query_tokens & target_tokens
    )

    return (
        len(matching_tokens)
        / len(query_tokens)
    )


def phrase_match(query_text, target_text):
    """
    Return 1.0 when the normalized query phrase occurs in the
    target text. Otherwise return 0.0.
    """

    normalized_query = normalize_for_matching(
        query_text
    )

    normalized_target = normalize_for_matching(
        target_text
    )

    if not normalized_query:
        return 0.0

    if normalized_query in normalized_target:
        return 1.0

    return 0.0


# ============================================================
# QUESTION ANALYSIS
# ============================================================

def fallback_question_analysis(question):
    """
    Basic fallback used if LLM-based question analysis fails.

    Technical-looking tokens are treated as possible entities.
    """

    tokens = tokenize(question)

    likely_entities = []

    for token in tokens:
        if (
            "-" in token
            or "." in token
            or "_" in token
            or any(character.isdigit() for character in token)
            or token.isupper()
        ):
            likely_entities.append(token)

    return {
        "entities": list(dict.fromkeys(likely_entities)),
        "intent": question,
        "keywords": tokens[:8]
    }


def analyze_question(question):
    """
    Analyze the user's question before searching.

    The LLM does not answer the question here.

    The LLM only extracts:

    - exact entities
    - requested intent
    - alternative search terms
    """

    if not ENABLE_LLM_QUERY_ANALYSIS:
        return fallback_question_analysis(question)

    prompt = f"""
You are a query analyzer for an engineering document
retrieval system.

Analyze the question and return:

1. entities:
   Exact names explicitly present in the question, including
   services, components, modules, products, protocols,
   classes, property names, error names, requirement IDs,
   operating systems and features.

2. intent:
   A short description of the requested information, such as:
   scalability, configuration, architecture, compatibility,
   root cause, troubleshooting, deployment, performance,
   security, workflow, retry handling or data flow.

3. keywords:
   Up to eight search terms or close technical alternatives
   that could occur in an engineering document.

Important rules:
- Preserve technical entity names exactly.
- Do not answer the question.
- Do not invent a service or component.
- Distinguish similarly named services.
- The word "scale" may mean a physical weighing scale or
  service scalability. Use the full question to identify
  the intended meaning.
- Return valid JSON only.

Question:
{question}

Return exactly this structure:
{{
  "entities": [],
  "intent": "",
  "keywords": []
}}
""".strip()

    try:
        response = ollama.chat(
            model=LLM_MODEL,
            messages=[
                {
                    "role": "user",
                    "content": prompt
                }
            ],
            format="json",
            options={
                "temperature": 0
            }
        )

        raw_content = response[
            "message"
        ]["content"]

        parsed = json.loads(raw_content)

        entities = parsed.get(
            "entities",
            []
        )

        intent = parsed.get(
            "intent",
            ""
        )

        keywords = parsed.get(
            "keywords",
            []
        )

        if not isinstance(entities, list):
            entities = []

        if not isinstance(keywords, list):
            keywords = []

        if not isinstance(intent, str):
            intent = ""

        return {
            "entities": [
                str(entity).strip()
                for entity in entities
                if str(entity).strip()
            ],
            "intent": intent.strip(),
            "keywords": [
                str(keyword).strip()
                for keyword in keywords[:8]
                if str(keyword).strip()
            ]
        }

    except Exception as exception:
        print()
        print(
            "WARNING: LLM question analysis failed."
        )
        print(exception)
        print(
            "Using fallback question analysis."
        )

        return fallback_question_analysis(
            question
        )


def build_search_text(question, analysis):
    """
    Create an expanded search query for vector and BM25
    retrieval.

    Generated keywords improve retrieval only. They are not
    factual evidence for the final answer.
    """

    search_parts = [
        question,
        *analysis.get("entities", []),
        analysis.get("intent", ""),
        *analysis.get("keywords", [])
    ]

    cleaned_parts = []

    for part in search_parts:
        if not part:
            continue

        cleaned_part = str(part).strip()

        if cleaned_part:
            cleaned_parts.append(
                cleaned_part
            )

    return " ".join(cleaned_parts)


# ============================================================
# HYBRID RETRIEVER
# ============================================================

class HybridRetriever:

    def __init__(self):
        """
        Load ChromaDB records and create an in-memory BM25
        index.

        For hundreds or several thousand chunks, this is
        suitable for the POC.
        """

        self.client = chromadb.PersistentClient(
            path=str(CHROMA_DIRECTORY)
        )

        try:
            self.collection = (
                self.client.get_collection(
                    name=COLLECTION_NAME
                )
            )

        except Exception as exception:
            raise RuntimeError(
                f'ChromaDB collection "{COLLECTION_NAME}" '
                f"was not found.\n"
                f"Run ingest.py first."
            ) from exception

        self.records = self._load_records()

        if not self.records:
            raise RuntimeError(
                "The ChromaDB collection is empty.\n"
                "Run ingest.py before retrieval."
            )

        print(
            f"Loaded ChromaDB records: "
            f"{len(self.records)}"
        )

        print("Building BM25 index...")

        tokenized_corpus = [
            tokenize(record["document"])
            for record in self.records
        ]

        self.bm25_index = BM25Okapi(
            tokenized_corpus,
            k1=1.2,
            b=0.75
        )

        print("BM25 index is ready.")

    # --------------------------------------------------------
    # LOAD RECORDS
    # --------------------------------------------------------

    def _load_records(self):
        """
        Load all stored documents and metadata.

        BM25 is built from these records.
        """

        stored_data = self.collection.get(
            include=[
                "documents",
                "metadatas"
            ]
        )

        record_ids = stored_data.get(
            "ids",
            []
        )

        documents = stored_data.get(
            "documents",
            []
        )

        metadatas = stored_data.get(
            "metadatas",
            []
        )

        records = []

        for record_id, document, metadata in zip(
            record_ids,
            documents,
            metadatas
        ):
            if not document:
                continue

            records.append({
                "id": record_id,
                "document": document,
                "metadata": metadata or {}
            })

        return records

    # --------------------------------------------------------
    # VECTOR SEARCH
    # --------------------------------------------------------

    def vector_search(self, search_text):
        """
        Embed the expanded search text and retrieve semantically
        similar chunks from ChromaDB.
        """

        embedding_response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=search_text
        )

        embeddings = embedding_response.get(
            "embeddings",
            []
        )

        if not embeddings:
            raise RuntimeError(
                "Ollama returned no query embedding."
            )

        query_embedding = embeddings[0]

        candidate_count = min(
            VECTOR_CANDIDATES,
            self.collection.count()
        )

        result = self.collection.query(
            query_embeddings=[
                query_embedding
            ],
            n_results=candidate_count,
            include=[
                "documents",
                "metadatas",
                "distances"
            ]
        )

        vector_results = []

        result_ids = result.get(
            "ids",
            [[]]
        )[0]

        result_documents = result.get(
            "documents",
            [[]]
        )[0]

        result_metadatas = result.get(
            "metadatas",
            [[]]
        )[0]

        result_distances = result.get(
            "distances",
            [[]]
        )[0]

        for rank, (
            record_id,
            document,
            metadata,
            distance
        ) in enumerate(
            zip(
                result_ids,
                result_documents,
                result_metadatas,
                result_distances
            ),
            start=1
        ):
            vector_results.append({
                "id": record_id,
                "document": document,
                "metadata": metadata or {},
                "vector_rank": rank,
                "vector_distance": float(
                    distance
                )
            })

        return vector_results

    # --------------------------------------------------------
    # BM25 SEARCH
    # --------------------------------------------------------

    def bm25_search(self, search_text):
        """
        Perform lexical retrieval using BM25.

        BM25 is particularly useful for exact engineering terms
        such as:

        - requirement IDs
        - service names
        - property names
        - exception names
        - operating system names
        """

        query_tokens = tokenize(search_text)

        if not query_tokens:
            return []

        scores = self.bm25_index.get_scores(
            query_tokens
        )

        ranked_indexes = sorted(
            range(len(self.records)),
            key=lambda index: scores[index],
            reverse=True
        )

        bm25_results = []

        for record_index in ranked_indexes:
            score = float(
                scores[record_index]
            )

            # Skip records having no lexical match.
            if score <= 0:
                continue

            record = self.records[
                record_index
            ]

            bm25_results.append({
                "id": record["id"],
                "document": record[
                    "document"
                ],
                "metadata": record[
                    "metadata"
                ],
                "bm25_rank": (
                    len(bm25_results) + 1
                ),
                "bm25_score": score
            })

            if (
                len(bm25_results)
                >= BM25_CANDIDATES
            ):
                break

        return bm25_results

    # --------------------------------------------------------
    # RECIPROCAL RANK FUSION
    # --------------------------------------------------------

    def reciprocal_rank_fusion(
        self,
        vector_results,
        bm25_results
    ):
        """
        Combine vector and BM25 ranking.

        RRF combines rank positions rather than incompatible
        raw score ranges.

        Formula:

            weight / (constant + rank)
        """

        fused_records = {}

        for result in vector_results:
            record_id = result["id"]

            fused_records[record_id] = {
                "id": record_id,
                "document": result[
                    "document"
                ],
                "metadata": result[
                    "metadata"
                ],
                "vector_rank": result[
                    "vector_rank"
                ],
                "vector_distance": result[
                    "vector_distance"
                ],
                "bm25_rank": None,
                "bm25_score": None,
                "rrf_score": (
                    VECTOR_WEIGHT
                    / (
                        RRF_CONSTANT
                        + result["vector_rank"]
                    )
                )
            }

        for result in bm25_results:
            record_id = result["id"]

            if record_id not in fused_records:
                fused_records[record_id] = {
                    "id": record_id,
                    "document": result[
                        "document"
                    ],
                    "metadata": result[
                        "metadata"
                    ],
                    "vector_rank": None,
                    "vector_distance": None,
                    "bm25_rank": None,
                    "bm25_score": None,
                    "rrf_score": 0.0
                }

            fused_records[record_id][
                "bm25_rank"
            ] = result["bm25_rank"]

            fused_records[record_id][
                "bm25_score"
            ] = result["bm25_score"]

            fused_records[record_id][
                "rrf_score"
            ] += (
                BM25_WEIGHT
                / (
                    RRF_CONSTANT
                    + result["bm25_rank"]
                )
            )

        ranked_results = sorted(
            fused_records.values(),
            key=lambda item: item[
                "rrf_score"
            ],
            reverse=True
        )

        return ranked_results[
            :FUSED_CANDIDATES
        ]

    # --------------------------------------------------------
    # GENERIC METADATA RERANKING
    # --------------------------------------------------------

    def metadata_rerank(
        self,
        candidates,
        analysis
    ):
        """
        Rerank candidates using automatically extracted query
        entities and intent.

        No component names are hardcoded.

        A section title matching both entity and intent receives
        a higher score than a body-only match.
        """

        entities = analysis.get(
            "entities",
            []
        )

        intent = analysis.get(
            "intent",
            ""
        )

        keywords = analysis.get(
            "keywords",
            []
        )

        keyword_text = " ".join(
            keywords
        )

        for candidate in candidates:
            metadata = candidate.get(
                "metadata",
                {}
            )

            section_title = metadata.get(
                "section_title",
                ""
            )

            section_id = metadata.get(
                "section_id",
                ""
            )

            source = metadata.get(
                "source",
                ""
            )

            document_text = candidate.get(
                "document",
                ""
            )

            entity_title_score = 0.0
            entity_body_score = 0.0
            entity_filename_score = 0.0
            exact_entity_title_score = 0.0

            for entity in entities:
                entity_title_score = max(
                    entity_title_score,
                    token_overlap(
                        entity,
                        section_title
                    )
                )

                entity_body_score = max(
                    entity_body_score,
                    token_overlap(
                        entity,
                        document_text
                    )
                )

                entity_filename_score = max(
                    entity_filename_score,
                    token_overlap(
                        entity,
                        source
                    )
                )

                exact_entity_title_score = max(
                    exact_entity_title_score,
                    phrase_match(
                        entity,
                        section_title
                    )
                )

            intent_title_score = token_overlap(
                intent,
                section_title
            )

            intent_body_score = token_overlap(
                intent,
                document_text
            )

            keyword_title_score = token_overlap(
                keyword_text,
                section_title
            )

            keyword_body_score = token_overlap(
                keyword_text,
                document_text
            )

            section_id_score = 0.0

            for entity in entities:
                section_id_score = max(
                    section_id_score,
                    token_overlap(
                        entity,
                        section_id
                    )
                )

            # Generic structural weighting.
            #
            # A matching section title has much more value than
            # one occurrence somewhere in a long section body.
            metadata_score = (
                0.12 * exact_entity_title_score
                + 0.10 * entity_title_score
                + 0.025 * entity_body_score
                + 0.025 * entity_filename_score
                + 0.10 * intent_title_score
                + 0.020 * intent_body_score
                + 0.040 * keyword_title_score
                + 0.010 * keyword_body_score
                + 0.030 * section_id_score
            )

            candidate[
                "entity_title_score"
            ] = entity_title_score

            candidate[
                "intent_title_score"
            ] = intent_title_score

            candidate[
                "metadata_score"
            ] = metadata_score

            candidate[
                "pre_rerank_score"
            ] = (
                candidate.get(
                    "rrf_score",
                    0.0
                )
                + metadata_score
            )

        ranked_candidates = sorted(
            candidates,
            key=lambda item: item[
                "pre_rerank_score"
            ],
            reverse=True
        )

        return ranked_candidates[
            :LLM_RERANK_CANDIDATES
        ]

    # --------------------------------------------------------
    # LLM RERANKING
    # --------------------------------------------------------

    def llm_rerank(
        self,
        question,
        candidates
    ):
        """
        Ask the local model to rank the final candidate set.

        The model is not asked to answer the question.

        The reranker only determines which candidate provides
        the most direct evidence.
        """

        if not ENABLE_LLM_RERANKING:
            return candidates

        if not candidates:
            return []

        candidate_blocks = []

        for candidate_number, candidate in enumerate(
            candidates,
            start=1
        ):
            metadata = candidate.get(
                "metadata",
                {}
            )

            content = candidate.get(
                "document",
                ""
            )[:RERANK_CONTENT_LIMIT]

            candidate_blocks.append(
                f"""
CANDIDATE {candidate_number}

Document:
{metadata.get("source", "Unknown")}

Document type:
{metadata.get("document_type", "Unknown")}

Section ID:
{metadata.get("section_id", "Unknown")}

Section title:
{metadata.get("section_title", "Unknown")}

Content:
{content}
""".strip()
            )

        candidates_text = (
            "\n\n"
            "----------------------------------------"
            "\n\n"
        ).join(candidate_blocks)

        prompt = f"""
You rerank search results from engineering documents.

Question:
{question}

Rank the candidates according to how directly the candidate
contains evidence needed to answer the exact question.

Rules:
1. Match the exact service, component, module, property,
   feature or error requested by the user.
2. Do not treat similarly named components as identical.
3. Prefer a section title matching both:
   - the requested entity
   - the requested topic or intent
4. Distinguish physical weighing scales from software service
   scalability.
5. A candidate merely mentioning an entity is less relevant
   than a candidate explaining the requested behavior.
6. Do not answer the question.
7. Do not use outside knowledge.
8. Include every candidate once.
9. Use relevance scores from 0.0 to 1.0.
10. Return valid JSON only.

Candidates:
{candidates_text}

Return exactly:
{{
  "ranking": [
    {{
      "candidate": 1,
      "relevance": 0.0,
      "reason": "Short explanation"
    }}
  ]
}}
""".strip()

        try:
            response = ollama.chat(
                model=LLM_MODEL,
                messages=[
                    {
                        "role": "user",
                        "content": prompt
                    }
                ],
                format="json",
                options={
                    "temperature": 0
                }
            )

            raw_content = response[
                "message"
            ]["content"]

            parsed = json.loads(raw_content)

            ranking = parsed.get(
                "ranking",
                []
            )

            if not isinstance(ranking, list):
                return candidates

            valid_ranking = []

            for item in ranking:
                try:
                    candidate_number = int(
                        item.get(
                            "candidate",
                            0
                        )
                    )

                    relevance = float(
                        item.get(
                            "relevance",
                            0.0
                        )
                    )

                except (
                    TypeError,
                    ValueError
                ):
                    continue

                if not (
                    1
                    <= candidate_number
                    <= len(candidates)
                ):
                    continue

                valid_ranking.append({
                    "candidate": candidate_number,
                    "relevance": max(
                        0.0,
                        min(1.0, relevance)
                    ),
                    "reason": str(
                        item.get(
                            "reason",
                            ""
                        )
                    )
                })

            valid_ranking.sort(
                key=lambda item: item[
                    "relevance"
                ],
                reverse=True
            )

            reranked_results = []
            used_candidate_indexes = set()

            for item in valid_ranking:
                candidate_index = (
                    item["candidate"] - 1
                )

                if (
                    candidate_index
                    in used_candidate_indexes
                ):
                    continue

                candidate_copy = dict(
                    candidates[candidate_index]
                )

                candidate_copy[
                    "reranker_relevance"
                ] = item["relevance"]

                candidate_copy[
                    "reranker_reason"
                ] = item["reason"]

                reranked_results.append(
                    candidate_copy
                )

                used_candidate_indexes.add(
                    candidate_index
                )

            # If the model omitted candidates, preserve those
            # candidates at the end in metadata-ranked order.
            for candidate_index, candidate in enumerate(
                candidates
            ):
                if (
                    candidate_index
                    not in used_candidate_indexes
                ):
                    candidate_copy = dict(
                        candidate
                    )

                    candidate_copy[
                        "reranker_relevance"
                    ] = None

                    candidate_copy[
                        "reranker_reason"
                    ] = (
                        "Candidate was not returned by "
                        "the LLM reranker."
                    )

                    reranked_results.append(
                        candidate_copy
                    )

            return reranked_results

        except Exception as exception:
            print()
            print(
                "WARNING: LLM reranking failed."
            )
            print(exception)
            print(
                "Using metadata ranking instead."
            )

            return candidates

    # --------------------------------------------------------
    # COMPLETE SEARCH PIPELINE
    # --------------------------------------------------------

    def search(self, question):
        """
        Complete hybrid retrieval pipeline.

        Returns all intermediate rankings for debugging and
        the final context chunks for answer generation.
        """

        if not question or not question.strip():
            raise ValueError(
                "Question cannot be empty."
            )

        question = question.strip()

        analysis = analyze_question(
            question
        )

        search_text = build_search_text(
            question,
            analysis
        )

        vector_results = self.vector_search(
            search_text
        )

        bm25_results = self.bm25_search(
            search_text
        )

        fused_results = (
            self.reciprocal_rank_fusion(
                vector_results,
                bm25_results
            )
        )

        metadata_results = (
            self.metadata_rerank(
                fused_results,
                analysis
            )
        )

        reranked_results = self.llm_rerank(
            question,
            metadata_results
        )

        final_results = reranked_results[
            :FINAL_CONTEXT_CHUNKS
        ]

        return {
            "question": question,
            "analysis": analysis,
            "search_text": search_text,
            "vector_results": vector_results,
            "bm25_results": bm25_results,
            "fused_results": fused_results,
            "metadata_results": metadata_results,
            "reranked_results": reranked_results,
            "final_results": final_results
        }


# ============================================================
# DEBUG PRINT FUNCTIONS
# ============================================================

def print_question_analysis(search_output):
    analysis = search_output["analysis"]

    print()
    print("=" * 70)
    print("QUESTION ANALYSIS")
    print("=" * 70)

    print(
        "Entities:",
        analysis.get("entities", [])
    )

    print(
        "Intent:",
        analysis.get("intent", "")
    )

    print(
        "Keywords:",
        analysis.get("keywords", [])
    )

    print()
    print("Expanded search text:")
    print(search_output["search_text"])


def print_vector_results(results, limit=10):
    print()
    print("=" * 70)
    print("VECTOR RANKING")
    print("=" * 70)

    for result in results[:limit]:
        metadata = result.get(
            "metadata",
            {}
        )

        print()
        print(
            f"Rank: {result['vector_rank']}"
        )

        print(
            f"Distance: "
            f"{result['vector_distance']:.6f}"
        )

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section_id')} | "
            f"{metadata.get('section_title')}"
        )

        print(
            result["document"][:500]
        )


def print_bm25_results(results, limit=10):
    print()
    print("=" * 70)
    print("BM25 RANKING")
    print("=" * 70)

    for result in results[:limit]:
        metadata = result.get(
            "metadata",
            {}
        )

        print()
        print(
            f"Rank: {result['bm25_rank']}"
        )

        print(
            f"Score: "
            f"{result['bm25_score']:.6f}"
        )

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section_id')} | "
            f"{metadata.get('section_title')}"
        )

        print(
            result["document"][:500]
        )


def print_fused_results(results):
    print()
    print("=" * 70)
    print("RRF FUSED RANKING")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):
        metadata = result.get(
            "metadata",
            {}
        )

        print()
        print(f"Rank: {rank}")

        print(
            f"RRF score: "
            f"{result.get('rrf_score', 0):.8f}"
        )

        print(
            f"Vector rank: "
            f"{result.get('vector_rank')}"
        )

        print(
            f"BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section_id')} | "
            f"{metadata.get('section_title')}"
        )


def print_final_results(results):
    print()
    print("=" * 70)
    print("FINAL RETRIEVAL RANKING")
    print("=" * 70)

    for rank, result in enumerate(
        results,
        start=1
    ):
        metadata = result.get(
            "metadata",
            {}
        )

        print()
        print(f"Final rank: {rank}")

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Document type: "
            f"{metadata.get('document_type')}"
        )

        print(
            f"Section ID: "
            f"{metadata.get('section_id')}"
        )

        print(
            f"Section title: "
            f"{metadata.get('section_title')}"
        )

        print(
            f"Vector rank: "
            f"{result.get('vector_rank')}"
        )

        print(
            f"BM25 rank: "
            f"{result.get('bm25_rank')}"
        )

        print(
            f"RRF score: "
            f"{result.get('rrf_score', 0):.8f}"
        )

        print(
            f"Metadata score: "
            f"{result.get('metadata_score', 0):.6f}"
        )

        print(
            f"LLM relevance: "
            f"{result.get('reranker_relevance')}"
        )

        print(
            f"Reranker reason: "
            f"{result.get('reranker_reason', '')}"
        )

        print()
        print(result["document"][:1000])


# ============================================================
# DIRECT TEST
# ============================================================

def main():
    """
    Allows retrieval.py to be tested independently before
    connecting it to ask.py.
    """

    print("=" * 70)
    print("HYBRID ENGINEERING DOCUMENT RETRIEVAL")
    print("=" * 70)

    print(
        f"ChromaDB path: {CHROMA_DIRECTORY}"
    )

    print(
        f"Collection: {COLLECTION_NAME}"
    )

    print(
        f"Embedding model: {EMBEDDING_MODEL}"
    )

    print(
        f"LLM model: {LLM_MODEL}"
    )

    print()

    retriever = HybridRetriever()

    while True:
        question = input(
            "\nAsk a question, or type exit: "
        ).strip()

        if question.lower() == "exit":
            break

        if not question:
            continue

        search_output = retriever.search(
            question
        )

        print_question_analysis(
            search_output
        )

        print_vector_results(
            search_output[
                "vector_results"
            ]
        )

        print_bm25_results(
            search_output[
                "bm25_results"
            ]
        )

        print_fused_results(
            search_output[
                "fused_results"
            ]
        )

        print_final_results(
            search_output[
                "final_results"
            ]
        )


if __name__ == "__main__":
    main()