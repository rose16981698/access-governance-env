from __future__ import annotations

import argparse
import json

from access_governance_env.baseline import benchmark_baseline_suite


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run the heuristic baseline across easy, medium, and hard tasks."
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Print the baseline scores as JSON only.",
    )
    args = parser.parse_args()

    scores = benchmark_baseline_suite()
    if args.json:
        print(json.dumps(scores, indent=2))
        return

    print("Baseline task scores:")
    for difficulty, score in scores.items():
        print(f"- {difficulty}: {score}")


if __name__ == "__main__":
    main()
