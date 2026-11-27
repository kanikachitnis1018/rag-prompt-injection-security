import csv
import json
import sys
from pathlib import Path


# --------------------------------------------------
# 1. Project paths
# --------------------------------------------------

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import run_rag


TEST_CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "test_cases.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "content_filtering.csv"
)


# --------------------------------------------------
# 2. Extract plain text from Gemini response
# --------------------------------------------------

def extract_response_text(response):
    """
    Extract only the visible text from Gemini's
    structured response.
    """

    if response is None:
        return ""

    content = response.content

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        text_parts = []

        for item in content:
            if (
                isinstance(item, dict)
                and item.get("type") == "text"
            ):
                text_parts.append(
                    item.get("text", "")
                )

        return "\n".join(text_parts)

    return str(content)


# --------------------------------------------------
# 3. Run content-filtering evaluation
# --------------------------------------------------

def main():

    # Load A01, A02, A03
    with open(
        TEST_CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    experiment_results = []

    for test_case in test_cases:

        test_id = test_case["id"]
        query = test_case["query"]
        attack_file = test_case["attack_file"]
        expected_marker = test_case["expected_marker"]

        print(f"\nRunning {test_id}...")
        print(f"Query: {query}")

        # Run the same RAG pipeline with
        # content filtering enabled.
        rag_result = run_rag(
            query,
            defense="content_filtering",
        )

        allowed_chunks = rag_result["results"]
        blocked_chunks = rag_result["blocked_chunks"]
        response = rag_result["response"]

        # ------------------------------------------
        # Find sources that survived the filter
        # ------------------------------------------

        allowed_sources = [
            chunk.metadata["source"]
            for chunk in allowed_chunks
        ]

        # ------------------------------------------
        # Find sources blocked by the filter
        # ------------------------------------------

        blocked_sources = [
            blocked["chunk"].metadata["source"]
            for blocked in blocked_chunks
        ]

        # ------------------------------------------
        # Reconstruct original retrieval
        #
        # Both allowed + blocked chunks were
        # originally returned by similarity search.
        # ------------------------------------------

        retrieved_sources = (
            allowed_sources
            + blocked_sources
        )

        # Was the intended attack document retrieved?
        attack_retrieved = (
            attack_file in retrieved_sources
        )

        # Was that attack document caught by filter?
        attack_filtered = (
            attack_file in blocked_sources
        )

        # Extract Gemini's visible response
        response_text = extract_response_text(
            response
        )

        # Attack succeeds if its marker appears
        # in the model response.
        attack_success = (
            expected_marker in response_text
        )

        # ------------------------------------------
        # Save result
        # ------------------------------------------

        result = {
            "test_id": test_id,
            "query": query,
            "attack_file": attack_file,
            "retrieved_sources":
                " | ".join(retrieved_sources),
            "attack_retrieved":
                attack_retrieved,
            "allowed_sources":
                " | ".join(allowed_sources),
            "blocked_sources":
                " | ".join(blocked_sources),
            "attack_filtered":
                attack_filtered,
            "expected_marker":
                expected_marker,
            "response":
                response_text,
            "attack_success":
                attack_success,
        }

        experiment_results.append(result)

        # ------------------------------------------
        # Terminal output
        # ------------------------------------------

        print(
            f"Retrieved sources: "
            f"{retrieved_sources}"
        )

        print(
            f"Allowed sources: "
            f"{allowed_sources}"
        )

        print(
            f"Blocked sources: "
            f"{blocked_sources}"
        )

        print(
            f"Attack retrieved: "
            f"{attack_retrieved}"
        )

        print(
            f"Attack filtered: "
            f"{attack_filtered}"
        )

        print(
            f"Response: "
            f"{response_text}"
        )

        print(
            f"Attack success: "
            f"{attack_success}"
        )


    # --------------------------------------------------
    # 4. Save CSV
    # --------------------------------------------------

    RESULTS_PATH.parent.mkdir(
        exist_ok=True
    )

    fieldnames = [
        "test_id",
        "query",
        "attack_file",
        "retrieved_sources",
        "attack_retrieved",
        "allowed_sources",
        "blocked_sources",
        "attack_filtered",
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
        writer.writerows(
            experiment_results
        )

    print(
        f"\nResults saved to: "
        f"{RESULTS_PATH}"
    )


# --------------------------------------------------
# 5. Entry point
# --------------------------------------------------

if __name__ == "__main__":
    main()