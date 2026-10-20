
import json
import sys
from pathlib import Path

import requests
from docx import Document
from pypdf import PdfReader

MODEL = "gemma3"
OLLAMA_URL = "http://localhost:11434/api/chat"


def read_srs(file_path):
    path = Path(file_path)
    suffix = path.suffix.lower()

    if suffix == ".pdf":
        reader = PdfReader(str(path))
        return "\n".join(page.extract_text() or "" for page in reader.pages)

    if suffix == ".docx":
        doc = Document(str(path))
        return "\n".join(p.text for p in doc.paragraphs)

    if suffix in {".txt", ".md"}:
        return path.read_text(encoding="utf-8")

    raise ValueError("Supported formats: PDF, DOCX, TXT, MD")


def ask_model(system_prompt, user_prompt):
    response = requests.post(
        OLLAMA_URL,
        json={
            "model": MODEL,
            "stream": False,
            "format": "json",
            "messages": [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ],
        },
        timeout=300,
    )
    response.raise_for_status()
    content = response.json()["message"]["content"]
    return json.loads(content)


def generate_test_cases(srs_text):
    system_prompt = """
You are a software developer test-case generation agent.

Use ONLY the requirements stated in the supplied SRS.
Do not invent product behavior or assume missing business rules.

Identify functional requirements and generate developer-level test cases:
- Happy paths
- Negative cases
- Boundary conditions
- Input validation
- Error handling
- State transitions, if specified

For every test case include:
- id
- requirement_reference (use the SRS ID if present)
- scenario
- preconditions
- test_steps (an array)
- test_data
- expected_result
- test_type

If behavior is unspecified, list it under open_questions.
Do not claim a requirement is covered unless a test maps to it.

Return JSON with this exact structure:
{
  "test_cases": [],
  "open_questions": [],
  "requirements_needing_clarification": []
}
"""

    result = ask_model(
        system_prompt,
        "Analyze this SRS and generate developer test cases:\n\n" + srs_text,
    )

    # Second pass: review the generated tests against the source.
    review_prompt = """
Review the proposed developer test cases against the original SRS.

Check for:
1. Unsupported assumptions or invented behavior.
2. Missing negative or boundary cases where requirements justify them.
3. Missing requirement references.
4. Duplicate or vague tests.
5. Expected results that are not supported by the SRS.

Correct the test cases and preserve the JSON schema:
test_cases, open_questions, requirements_needing_clarification.
Return JSON only.
"""

    reviewed = ask_model(
        review_prompt,
        "ORIGINAL SRS:\n"
        + srs_text
        + "\n\nPROPOSED TEST CASES:\n"
        + json.dumps(result, ensure_ascii=False),
    )

    return reviewed


def main():
    if len(sys.argv) != 2:
        print("Usage: python srs_test_agent.py path/to/requirements.pdf")
        return

    file_path = Path(sys.argv[1])
    srs_text = read_srs(file_path)

    if not srs_text.strip():
        raise ValueError("No readable text found in the SRS.")

    print("Generating and reviewing test cases...")
    result = generate_test_cases(srs_text)

    output_path = file_path.with_name(file_path.stem + "_test_cases.json")
    output_path.write_text(
        json.dumps(result, indent=2, ensure_ascii=False),
        encoding="utf-8",
    )

    print(f"Done! Output saved to: {output_path}")


if __name__ == "__main__":
    main()
