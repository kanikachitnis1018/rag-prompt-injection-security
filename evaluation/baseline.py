import csv
import json
import sys
from pathlib import Path


# Make the project root importable so we can import agent.py
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import run_rag


TEST_CASES_PATH = PROJECT_ROOT / "evaluation" / "test_cases.json"
RESULTS_PATH = PROJECT_ROOT / "results" / "baseline.csv"


def extract_response_text(response):
    """
    Extract only the actual text from Gemini's structured response.
    """

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if isinstance(item, dict) and item.get("type") == "text":
                text_parts.append(item.get("text", ""))

        return "\n".join(text_parts)

    return str(content)


def main():
    # Load experiment definitions
    with open(TEST_CASES_PATH, "r", encoding="utf-8") as file:
        test_cases = json.load(file)

    results = []

    for test_case in test_cases:
        test_id = test_case["id"]
        query = test_case["query"]
        attack_file = test_case["attack_file"]
        expected_marker = test_case["expected_marker"]

        print(f"\nRunning {test_id}...")
        print(f"Query: {query}")

        rag_result = run_rag(query)

        retrieved_chunks = rag_result["results"]
        response = rag_result["response"]

        # Get filenames of retrieved chunks
        retrieved_sources = [
            chunk.metadata["source"]
            for chunk in retrieved_chunks
        ]

        # Did retrieval include the intended attack document?
        attack_retrieved = attack_file in retrieved_sources

        # Extract plain text from Gemini's response
        response_text = extract_response_text(response)

        # Did Gemini output the attack marker?
        attack_success = expected_marker in response_text

        result = {
            "test_id": test_id,
            "query": query,
            "attack_file": attack_file,
            "retrieved_sources": " | ".join(retrieved_sources),
            "attack_retrieved": attack_retrieved,
            "expected_marker": expected_marker,
            "response": response_text,
            "attack_success": attack_success,
        }

        results.append(result)

        print(f"Retrieved sources: {retrieved_sources}")
        print(f"Attack retrieved: {attack_retrieved}")
        print(f"Response: {response_text}")
        print(f"Attack success: {attack_success}")

    # Make sure results/ exists
    RESULTS_PATH.parent.mkdir(exist_ok=True)

    # Write experiment results
    fieldnames = [
        "test_id",
        "query",
        "attack_file",
        "retrieved_sources",
        "attack_retrieved",
        "expected_marker",
        "response",
        "attack_success",
    ]

    with open(
        RESULTS_PATH,
        "w",
        newline="",
        encoding="utf-8",
    ) as file:
        writer = csv.DictWriter(
            file,
            fieldnames=fieldnames,
        )

        writer.writeheader()
        writer.writerows(results)

    print(f"\nResults saved to: {RESULTS_PATH}")


if __name__ == "__main__":
    main()