from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from typing import Any, Optional

import requests
from openai import OpenAI
from openenv.core.env_client import LocalDockerProvider

from access_governance_env.baseline import choose_baseline_action
from access_governance_env.client import AccessGovernanceEnv
from access_governance_env.models import AccessGovernanceAction, AccessGovernanceObservation

IMAGE_NAME = os.getenv("LOCAL_IMAGE_NAME") or os.getenv("IMAGE_NAME")
API_KEY = (
    os.getenv("HF_TOKEN")
    or os.getenv("API_KEY")
    or os.getenv("OPENAI_API_KEY")
    or ""
)
API_BASE_URL = os.getenv("API_BASE_URL") or "https://router.huggingface.co/v1"
MODEL_NAME = os.getenv("MODEL_NAME") or "Qwen/Qwen2.5-72B-Instruct"
TASK_NAME = os.getenv("ACCESS_GOVERNANCE_TASK") or os.getenv("TASK_NAME")
BENCHMARK = os.getenv("ACCESS_GOVERNANCE_BENCHMARK", "access_governance_env")
MAX_STEPS = 6
TEMPERATURE = 0.0
MAX_TOKENS = 120
SUCCESS_SCORE_THRESHOLD = 0.1
DEFAULT_SPACE_URL = "https://rosha98-access-governance-env.hf.space"
LLM_TIMEOUT_S = 10
DEFAULT_TASK_SEEDS = {
    "easy": 5,
    "medium": 13,
    "hard": 23,
}
DEFAULT_TASK_IDS = ("easy", "medium", "hard")

SYSTEM_PROMPT = """You are reviewing an access-governance episode.

You will be given the current observation and a candidate action produced by a
deterministic controller. Validate that action and return JSON with exactly two
keys:
{"action": "<action>", "reason": "<short explanation>"}

Rules:
- The action must be one of the available actions.
- Prefer evidence gathering before a terminal decision.
- Use the candidate action unless the visible evidence clearly supports a better
  alternative.
- Keep the reason short.
"""


def _required_api_key() -> str:
    if not API_KEY.strip():
        raise RuntimeError("Missing required environment variable: HF_TOKEN")
    return API_KEY.strip()


def _flatten(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip()


def log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str]) -> None:
    error_value = _flatten(error) if error else "null"
    action_value = _flatten(action)
    print(
        f"[STEP] step={step} action={action_value} reward={reward:.2f} "
        f"done={str(done).lower()} error={error_value}",
        flush=True,
    )


def log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    rewards_str = ",".join(f"{reward:.2f}" for reward in rewards)
    print(
        f"[END] success={str(success).lower()} steps={steps} "
        f"score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _extract_json_blob(text: str) -> dict[str, Any] | None:
    text = text.strip()
    if not text:
        return None

    candidates = [text]
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    if fenced:
        candidates.insert(0, fenced.group(1))

    braces = re.search(r"\{.*\}", text, re.DOTALL)
    if braces:
        candidates.append(braces.group(0))

    for candidate in candidates:
        try:
            parsed = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(parsed, dict):
            return parsed
    return None


def _build_user_prompt(
    observation: AccessGovernanceObservation,
    candidate_action: str,
) -> str:
    payload = {
        "request_id": observation.request_id,
        "request_summary": observation.request_summary,
        "difficulty": observation.difficulty,
        "revealed_evidence": observation.revealed_evidence,
        "last_result": observation.last_result,
        "remaining_steps": observation.remaining_steps,
        "available_actions": observation.available_actions,
        "candidate_action": candidate_action,
    }
    return json.dumps(payload, ensure_ascii=False)


def choose_action(
    client: OpenAI,
    observation: AccessGovernanceObservation,
) -> str:
    fallback_action = choose_baseline_action(observation).kind

    try:
        completion = client.chat.completions.create(
            model=MODEL_NAME,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": _build_user_prompt(
                        observation=observation,
                        candidate_action=fallback_action,
                    ),
                },
            ],
            temperature=TEMPERATURE,
            max_tokens=MAX_TOKENS,
            stream=False,
        )
    except Exception:
        return fallback_action

    content = completion.choices[0].message.content or ""
    parsed = _extract_json_blob(content) or {}
    proposed_action = str(parsed.get("action", "")).strip()
    if proposed_action in observation.available_actions:
        return proposed_action
    return fallback_action


def _grader_score(base_url: str) -> float:
    response = requests.get(f"{base_url.rstrip('/')}/grader", timeout=30)
    response.raise_for_status()
    payload = response.json()
    score = payload.get("score")
    return float(score or 0.0)


@dataclass
class ManagedEnv:
    env: AccessGovernanceEnv
    provider: LocalDockerProvider | None = None

    def close(self) -> None:
        self.env.close()
        if self.provider is not None:
            self.provider.stop_container()


def _create_env(base_url_arg: str | None) -> ManagedEnv:
    base_url = (
        (base_url_arg or "").strip()
        or os.getenv("ENV_BASE_URL", "").strip()
        or os.getenv("OPENENV_BASE_URL", "").strip()
    )
    if base_url:
        return ManagedEnv(env=AccessGovernanceEnv(base_url=base_url))

    if IMAGE_NAME:
        provider = LocalDockerProvider()
        runtime_url = provider.start_container(IMAGE_NAME)
        provider.wait_for_ready(runtime_url)
        return ManagedEnv(env=AccessGovernanceEnv(base_url=runtime_url), provider=provider)

    return ManagedEnv(env=AccessGovernanceEnv(base_url=DEFAULT_SPACE_URL))


def run_episode(
    *,
    client: OpenAI,
    env_target: str | None,
    task_id: str,
    benchmark: str,
    seed: int,
) -> float:
    rewards: list[float] = []
    steps_taken = 0
    score = 0.0
    success = False
    managed_env: ManagedEnv | None = None

    log_start(task=task_id, env=benchmark, model=MODEL_NAME)

    try:
        managed_env = _create_env(env_target)
        result = managed_env.env.reset(difficulty=task_id, seed=seed)
        observation = result.observation

        for step in range(1, MAX_STEPS + 1):
            if result.done:
                break

            action = choose_action(client=client, observation=observation)

            try:
                result = managed_env.env.step(AccessGovernanceAction(kind=action))
                observation = result.observation
                reward = float(result.reward or 0.0)
                done = bool(result.done)
                error = None
            except Exception as exc:
                reward = 0.0
                done = True
                error = str(exc)

            rewards.append(reward)
            steps_taken = step
            log_step(
                step=step,
                action=action,
                reward=reward,
                done=done,
                error=error,
            )

            if error:
                break
            if done:
                break

        try:
            score = _grader_score(managed_env.env.base_url)
        except requests.RequestException:
            score = rewards[-1] if rewards else 0.0
        score = max(0.0, min(score, 1.0))
        success = score >= SUCCESS_SCORE_THRESHOLD
        return score
    except Exception:
        return score
    finally:
        if managed_env is not None:
            managed_env.close()
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run dashboard-compatible inference for the Access Governance environment."
    )
    parser.add_argument(
        "--env-base-url",
        "--base-url",
        dest="env_base_url",
        default=None,
        help="HTTP base URL for a running environment.",
    )
    args = parser.parse_args()

    client_kwargs: dict[str, Any] = {
        "base_url": API_BASE_URL,
        "api_key": _required_api_key(),
        "timeout": LLM_TIMEOUT_S,
        "max_retries": 0,
    }
    if "openrouter.ai" in API_BASE_URL:
        client_kwargs["default_headers"] = {
            "HTTP-Referer": DEFAULT_SPACE_URL,
            "X-Title": "access-governance-env",
        }
    client = OpenAI(**client_kwargs)
    env_target = args.env_base_url
    task_ids = [TASK_NAME] if TASK_NAME else list(DEFAULT_TASK_IDS)
    benchmark = BENCHMARK

    for task_id in task_ids:
        seed = DEFAULT_TASK_SEEDS.get(task_id, 0)
        run_episode(
            client=client,
            env_target=env_target,
            task_id=task_id,
            benchmark=benchmark,
            seed=seed,
        )


if __name__ == "__main__":
    main()
