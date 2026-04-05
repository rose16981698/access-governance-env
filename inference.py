from __future__ import annotations

import argparse
import json
import os
import re
import sys
from typing import Any

import requests
from openai import OpenAI

from access_governance_env.baseline import choose_baseline_action
from access_governance_env.client import AccessGovernanceEnv
from access_governance_env.models import AccessGovernanceAction, AccessGovernanceObservation


DEFAULT_TASK_SEEDS = {
    "easy": 5,
    "medium": 13,
    "hard": 23,
}

SYSTEM_PROMPT = """You are an enterprise access governance agent operating inside a deterministic OpenEnv task.

You are given a candidate action chosen by the controller. Your job is to validate it and explain it briefly.

Use this operating policy:
- Prefer evidence gathering before making a final decision.
- Prioritize missing lookups in this order when evidence is still sparse:
  1. inspect_resource_metadata
  2. inspect_requester_profile
  3. inspect_business_context
  4. inspect_access_history
  5. inspect_policy_rules
- Deny if the visible evidence shows any hard blocker such as separation-of-duties conflict, missing or denied manager approval, missing training for high/critical access, or restricted contractor elevated access.
- Escalate if the visible evidence shows emergency access, risky prior access history, a review-exception case, or critical access for a new employee.
- Approve only when the visible evidence supports a clean auto-approval path.

Respond with JSON only, with exactly these keys:
{"action": "<repeat the candidate action>", "reason": "<short explanation>"}
"""


def _stdout_event(tag: str, payload: dict[str, Any]) -> None:
    print(f"{tag} {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}")
    sys.stdout.flush()


def _required_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"Missing required environment variable: {name}")
    return value


def _build_llm_client() -> tuple[OpenAI, str, str]:
    api_base_url = _required_env("API_BASE_URL")
    model_name = _required_env("MODEL_NAME")
    api_key = os.environ.get("HF_TOKEN", "").strip() or os.environ.get(
        "OPENAI_API_KEY", ""
    ).strip()
    if not api_key:
        raise RuntimeError("Missing required environment variable: HF_TOKEN")

    client_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "base_url": api_base_url,
        "timeout": 60,
    }
    if "openrouter.ai" in api_base_url:
        client_kwargs["default_headers"] = {
            "HTTP-Referer": os.environ.get(
                "APP_URL",
                "https://huggingface.co/spaces/rosha98/access-governance-env",
            ),
            "X-Title": "access-governance-env",
        }

    return OpenAI(**client_kwargs), api_base_url, model_name


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    fenced_match = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced_match:
        candidates.insert(0, fenced_match.group(1))

    brace_match = re.search(r"\{.*\}", text, re.DOTALL)
    if brace_match:
        candidates.append(brace_match.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _choose_action_with_llm(
    *,
    client: OpenAI,
    model_name: str,
    observation: AccessGovernanceObservation,
    fallback_action: str,
) -> tuple[str, str, str, str]:
    user_payload = {
        "request_id": observation.request_id,
        "request_summary": observation.request_summary,
        "difficulty": observation.difficulty,
        "last_result": observation.last_result,
        "remaining_steps": observation.remaining_steps,
        "available_actions": observation.available_actions,
        "revealed_evidence": observation.revealed_evidence,
        "candidate_action": fallback_action,
    }

    try:
        response = client.chat.completions.create(
            model=model_name,
            temperature=0,
            max_tokens=120,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(user_payload, ensure_ascii=False)},
            ],
        )
        content = response.choices[0].message.content or ""
    except Exception as exc:
        return (
            fallback_action,
            fallback_action,
            f"fallback_llm_error:{type(exc).__name__}",
            str(exc),
        )

    parsed = _extract_json_blob(content) or {}
    model_action = str(parsed.get("action", "")).strip() or fallback_action
    reason = str(parsed.get("reason", "")).strip() or "model_reason_missing"
    return fallback_action, model_action, reason, content


def _grader_score(env_base_url: str) -> dict[str, Any]:
    response = requests.get(f"{env_base_url.rstrip('/')}/grader", timeout=30)
    response.raise_for_status()
    return response.json()


def _task_ids(env_base_url: str) -> list[str]:
    response = requests.get(f"{env_base_url.rstrip('/')}/tasks", timeout=30)
    response.raise_for_status()
    payload = response.json()
    task_entries = payload.get("tasks", [])
    return [entry["id"] for entry in task_entries if "id" in entry]


def run_episode(
    *,
    env: AccessGovernanceEnv,
    llm_client: OpenAI,
    model_name: str,
    env_base_url: str,
    task_id: str,
    seed: int,
) -> float:
    step_index = 0
    reset_result = env.reset(difficulty=task_id, seed=seed)
    observation = reset_result.observation

    _stdout_event(
        "[STEP]",
        {
            "phase": "task_start",
            "task_id": task_id,
            "seed": seed,
            "request_id": observation.request_id,
            "difficulty": observation.difficulty,
        },
    )

    while not observation.done:
        step_index += 1
        fallback_action = choose_baseline_action(observation).kind
        action_kind, model_action, reason, raw_model_output = _choose_action_with_llm(
            client=llm_client,
            model_name=model_name,
            observation=observation,
            fallback_action=fallback_action,
        )

        step_result = env.step(AccessGovernanceAction(kind=action_kind))
        observation = step_result.observation

        _stdout_event(
            "[STEP]",
            {
                "phase": "action",
                "task_id": task_id,
                "seed": seed,
                "step_index": step_index,
                "action": action_kind,
                "model_action": model_action,
                "reason": reason,
                "fallback_action": fallback_action,
                "reward": observation.reward,
                "done": observation.done,
                "last_result": observation.last_result,
                "remaining_steps": observation.remaining_steps,
                "raw_model_output": raw_model_output,
            },
        )

    grader = _grader_score(env_base_url)
    score = float(grader.get("score") or 0.0)
    _stdout_event(
        "[STEP]",
        {
            "phase": "task_end",
            "task_id": task_id,
            "seed": seed,
            "score": score,
            "done": grader.get("done"),
            "score_breakdown": grader.get("score_breakdown"),
        },
    )
    return score


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run dashboard-compatible inference against the Access Governance "
            "environment."
        )
    )
    parser.add_argument(
        "--env-base-url",
        "--base-url",
        dest="env_base_url",
        default=(
            os.environ.get("ENV_BASE_URL")
            or os.environ.get("OPENENV_BASE_URL")
            or "http://localhost:8000"
        ),
        help="Environment base URL.",
    )
    args = parser.parse_args()

    llm_client, api_base_url, model_name = _build_llm_client()
    env_base_url = args.env_base_url.rstrip("/")
    task_ids = _task_ids(env_base_url)
    if not task_ids:
        raise RuntimeError("No tasks were returned by /tasks")

    _stdout_event(
        "[START]",
        {
            "env_base_url": env_base_url,
            "api_base_url": api_base_url,
            "model_name": model_name,
            "tasks": task_ids,
        },
    )

    scores: dict[str, float] = {}
    with AccessGovernanceEnv(base_url=env_base_url) as env:
        for task_id in task_ids:
            seed = DEFAULT_TASK_SEEDS.get(task_id, 0)
            scores[task_id] = run_episode(
                env=env,
                llm_client=llm_client,
                model_name=model_name,
                env_base_url=env_base_url,
                task_id=task_id,
                seed=seed,
            )

    average_score = round(sum(scores.values()) / len(scores), 4)
    _stdout_event(
        "[END]",
        {
            "scores": scores,
            "average_score": average_score,
            "task_count": len(scores),
        },
    )


if __name__ == "__main__":
    main()
