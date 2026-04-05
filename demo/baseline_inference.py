from __future__ import annotations

import argparse
import json

from access_governance_env.baseline import run_baseline_episode
from access_governance_env.server.environment import AccessGovernanceEnvironment


def main():
    parser = argparse.ArgumentParser(
        description="Run the heuristic baseline on the Access Governance environment."
    )
    parser.add_argument(
        "--difficulty",
        choices=["easy", "medium", "hard"],
        default="medium",
        help="Difficulty tier to sample.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Seed used for deterministic case generation.",
    )
    args = parser.parse_args()

    env = AccessGovernanceEnvironment()
    observation = env.reset_for_demo(difficulty=args.difficulty, seed=args.seed)
    transcript, final_observation = run_baseline_episode(env, observation=observation)

    print("Request summary:")
    print(final_observation.request_summary)
    print("\nTranscript:")
    for step in transcript:
        print(
            json.dumps(
                {
                    "action": step.action,
                    "last_result": step.last_result,
                    "reward": step.reward,
                    "done": step.done,
                }
            )
        )

    print("\nFinal score breakdown:")
    print(json.dumps(final_observation.score_breakdown or {}, indent=2))


if __name__ == "__main__":
    main()
