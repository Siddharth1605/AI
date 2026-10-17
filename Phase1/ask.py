import ollama

from config import LLM_MODEL
from retrieval import HybridRetriever


def print_question_analysis(search_output):
    analysis = search_output["analysis"]

    print("\nQUESTION ANALYSIS")
    print("=================")

    print(
        "Entities:",
        analysis["entities"]
    )

    print(
        "Intent:",
        analysis["intent"]
    )

    print(
        "Keywords:",
        analysis["keywords"]
    )

    print(
        "Expanded search:",
        search_output["search_text"]
    )


def print_vector_ranking(results):
    print("\nVECTOR RANKING")
    print("==============")

    for result in results[:10]:
        metadata = result["metadata"]

        print(
            f"\nRank: "
            f"{result['vector_rank']}"
        )

        print(
            f"Distance: "
            f"{result['vector_distance']:.4f}"
        )

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section_id')} - "
            f"{metadata.get('section_title')}"
        )

        print(
            result["document"][:350]
        )


def print_bm25_ranking(results):
    print("\nBM25 RANKING")
    print("============")

    for result in results[:10]:
        metadata = result["metadata"]

        print(
            f"\nRank: "
            f"{result['bm25_rank']}"
        )

        print(
            f"Score: "
            f"{result['bm25_score']:.4f}"
        )

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section_id')} - "
            f"{metadata.get('section_title')}"
        )

        print(
            result["document"][:350]
        )


def print_final_ranking(results):
    print("\nFINAL RANKING")
    print("=============")

    for rank, result in enumerate(
        results,
        start=1
    ):
        metadata = result["metadata"]

        print(f"\nFinal rank: {rank}")

        print(
            f"Document: "
            f"{metadata.get('source')}"
        )

        print(
            f"Section: "
            f"{metadata.get('section_id')} - "
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
            f"{result.get('rrf_score', 0):.6f}"
        )

        print(
            f"Metadata score: "
            f"{result.get('metadata_score', 0):.6f}"
        )

        print(
            f"Reranker relevance: "
            f"{result.get('reranker_relevance')}"
        )

        print(
            f"Reranker reason: "
            f"{result.get('reranker_reason', '')}"
        )

        print(result["document"][:700])


def generate_answer(
    question,
    retrieved_results
):
    context_sections = []

    for source_number, result in enumerate(
        retrieved_results,
        start=1
    ):
        metadata = result["metadata"]

        context_sections.append(
            f"""
[SOURCE {source_number}]
Document: {metadata.get("source", "Unknown")}
Document type: {metadata.get("document_type", "Unknown")}
Section ID: {metadata.get("section_id", "Unknown")}
Section title: {metadata.get("section_title", "Unknown")}

Content:
{result["document"]}
""".strip()
        )

    context = "\n\n---\n\n".join(
        context_sections
    )

    prompt = f"""
You are an engineering knowledge assistant.

Answer the question using only the supplied sources.

Rules:
1. Identify the exact component, service, feature or
   property requested by the user.
2. Do not mix information from similarly named but
   different components.
3. Prefer the source whose section title most directly
   matches the question.
4. Do not use outside knowledge.
5. Preserve technical names, property names, requirement
   IDs and default values exactly.
6. Cite factual statements using [SOURCE 1],
   [SOURCE 2], and so on.
7. If the supplied sources do not contain the answer,
   respond exactly:
   Information not found in the available documents.
8. Give a concise technical explanation.

Question:
{question}

Sources:
{context}

Answer:
""".strip()

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ],
        options={
            "temperature": 0
        }
    )

    return response["message"]["content"]


def main():
    print("Loading hybrid retrieval system...")

    retriever = HybridRetriever()

    print(
        f"Loaded chunks: "
        f"{len(retriever.records)}"
    )

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

        print_vector_ranking(
            search_output[
                "vector_results"
            ]
        )

        print_bm25_ranking(
            search_output[
                "bm25_results"
            ]
        )

        print_final_ranking(
            search_output[
                "final_results"
            ]
        )

        answer = generate_answer(
            question=question,
            retrieved_results=search_output[
                "final_results"
            ]
        )

        print("\nANSWER")
        print("======")
        print(answer)


if __name__ == "__main__":
    main()
