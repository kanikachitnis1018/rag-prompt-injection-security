import csv
import json
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from agent import run_rag


TEST_CASES_PATH = (
    PROJECT_ROOT
    / "evaluation"
    / "benign_test_cases.json"
)

RESULTS_PATH = (
    PROJECT_ROOT
    / "results"
    / "benign_utility.csv"
)


DEFENSES = [
    "none",
    "context_separation",
    "content_filtering",
    "combined",
]


def extract_response_text(response):
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


def main():
    with open(
        TEST_CASES_PATH,
        "r",
        encoding="utf-8",
    ) as file:
        test_cases = json.load(file)

    experiment_results = []

    for defense in DEFENSES:

        print(
            f"\n==============================\n"
            f"Testing defense: {defense}\n"
            f"=============================="
        )

        for test_case in test_cases:

            test_id = test_case["id"]
            query = test_case["query"]
            expected_source = (
                test_case["expected_source"]
            )

            print(f"\nRunning {test_id}...")
            print(f"Query: {query}")

            # No adversarial document is added.
            rag_result = run_rag(
                query=query,
                defense=defense,
                attack_file=None,
            )

            retrieved_chunks = (
                rag_result["original_results"]
            )

            allowed_chunks = (
                rag_result["results"]
            )

            blocked_chunks = (
                rag_result["blocked_chunks"]
            )

            response = rag_result["response"]

            retrieved_sources = [
                chunk.metadata["source"]
                for chunk in retrieved_chunks
            ]

            allowed_sources = [
                chunk.metadata["source"]
                for chunk in allowed_chunks
            ]

            blocked_sources = [
                blocked["chunk"].metadata["source"]
                for blocked in blocked_chunks
            ]

            expected_source_retrieved = (
                expected_source
                in retrieved_sources
            )

            expected_source_blocked = (
                expected_source
                in blocked_sources
            )

            response_text = (
                extract_response_text(
                    response
                )
            )

            response_generated = (
                bool(response_text.strip())
            )

            result = {
                "test_id": test_id,
                "defense": defense,
                "query": query,
                "expected_source":
                    expected_source,
                "retrieved_sources":
                    " | ".join(
                        retrieved_sources
                    ),
                "expected_source_retrieved":
                    expected_source_retrieved,
                "allowed_sources":
                    " | ".join(
                        allowed_sources
                    ),
                "blocked_sources":
                    " | ".join(
                        blocked_sources
                    ),
                "expected_source_blocked":
                    expected_source_blocked,
                "response_generated":
                    response_generated,
                "response":
                    response_text,
            }

            experiment_results.append(
                result
            )

            print(
                f"Retrieved: "
                f"{retrieved_sources}"
            )

            print(
                f"Blocked: "
                f"{blocked_sources}"
            )

            print(
                f"Expected source retrieved: "
                f"{expected_source_retrieved}"
            )

            print(
                f"Expected source blocked: "
                f"{expected_source_blocked}"
            )

            print(
                f"Response: "
                f"{response_text}"
            )

    RESULTS_PATH.parent.mkdir(
        exist_ok=True
    )

    fieldnames = [
        "test_id",
        "defense",
        "query",
        "expected_source",
        "retrieved_sources",
        "expected_source_retrieved",
        "allowed_sources",
        "blocked_sources",
        "expected_source_blocked",
        "response_generated",
        "response",
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


if __name__ == "__main__":
    main()