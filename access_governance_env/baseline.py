from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from .models import AccessGovernanceAction, AccessGovernanceObservation


FIXED_LOOKUP_ORDER: tuple[str, ...] = (
    "inspect_resource_metadata",
    "inspect_requester_profile",
    "inspect_business_context",
    "inspect_access_history",
    "inspect_policy_rules",
)
SCOPE_LEVELS = {"read": 1, "write": 2, "admin": 3}
AUTO_APPROVE_MATRIX = {
    "backend_engineer": {"code_repo": "write", "feature_flag_admin": "read"},
    "data_scientist": {"customer_warehouse": "read"},
    "sre": {"prod_db": "read", "feature_flag_admin": "admin"},
    "security_engineer": {"secrets_manager": "read", "code_repo": "read"},
    "product_manager": {"feature_flag_admin": "read"},
    "contractor_qa": {"code_repo": "read"},
}
REVIEW_ELIGIBLE_EXCEPTIONS = {
    ("backend_engineer", "feature_flag_admin", "write"),
    ("backend_engineer", "prod_db", "read"),
    ("sre", "prod_db", "write"),
    ("security_engineer", "secrets_manager", "admin"),
    ("product_manager", "customer_warehouse", "read"),
}
TASK_BENCHMARK_SEEDS = {
    "easy": (5, 7, 11),
    "medium": (13, 17, 19),
    "hard": (23, 29, 31),
}


@dataclass
class BaselineStep:
    action: str
    reward: float | None
    done: bool
    last_result: str


def _visible_bucket(
    observation: AccessGovernanceObservation, bucket: str
) -> dict[str, Any] | None:
    return observation.revealed_evidence.get(bucket)


def _is_visible_hard_deny(observation: AccessGovernanceObservation) -> bool:
    access = _visible_bucket(observation, "inspect_access_history") or {}
    context = _visible_bucket(observation, "inspect_business_context") or {}
    requester = _visible_bucket(observation, "inspect_requester_profile") or {}
    resource = _visible_bucket(observation, "inspect_resource_metadata") or {}

    if access.get("separation_of_duties_conflict") is True:
        return True
    if context.get("manager_approval") in {"pending", "denied"}:
        return True
    if (
        resource.get("resource_sensitivity") in {"high", "critical"}
        and requester.get("training_status") == "missing"
    ):
        return True
    if (
        requester.get("employment_type") == "contractor"
        and resource.get("requested_scope") in {"write", "admin"}
        and resource.get("requested_resource")
        in {"prod_db", "feature_flag_admin", "secrets_manager"}
    ):
        return True
    return False


def _can_visible_auto_approve(observation: AccessGovernanceObservation) -> bool:
    requester = _visible_bucket(observation, "inspect_requester_profile") or {}
    resource = _visible_bucket(observation, "inspect_resource_metadata") or {}
    context = _visible_bucket(observation, "inspect_business_context") or {}
    policy = _visible_bucket(observation, "inspect_policy_rules") or {}
    access = _visible_bucket(observation, "inspect_access_history") or {}

    if not requester or not resource or not context or not policy:
        return False
    if _is_visible_hard_deny(observation):
        return False
    if access.get("prior_access_history") in {
        "prior_denial",
        "overprovisioned_cleanup_pending",
    }:
        return False
    if context.get("emergency_access") is True:
        return False

    role = requester["requester_role"]
    requested_resource = resource["requested_resource"]
    requested_scope = resource["requested_scope"]
    allowed_scope = AUTO_APPROVE_MATRIX.get(role, {}).get(requested_resource)
    if allowed_scope is None:
        return False
    return SCOPE_LEVELS[requested_scope] <= SCOPE_LEVELS[allowed_scope]


def _visible_escalation_signal(observation: AccessGovernanceObservation) -> bool:
    requester = _visible_bucket(observation, "inspect_requester_profile") or {}
    resource = _visible_bucket(observation, "inspect_resource_metadata") or {}
    context = _visible_bucket(observation, "inspect_business_context") or {}
    access = _visible_bucket(observation, "inspect_access_history") or {}

    if context.get("emergency_access") is True:
        return True
    if access.get("prior_access_history") in {
        "prior_denial",
        "overprovisioned_cleanup_pending",
    }:
        return True
    if (
        requester.get("requester_role"),
        resource.get("requested_resource"),
        resource.get("requested_scope"),
    ) in REVIEW_ELIGIBLE_EXCEPTIONS:
        return True
    if (
        resource.get("resource_sensitivity") == "critical"
        and requester.get("tenure_months", 999) < 6
    ):
        return True
    return False


def choose_baseline_action(
    observation: AccessGovernanceObservation,
) -> AccessGovernanceAction:
    if observation.done:
        raise RuntimeError("Cannot choose an action for a finished episode.")

    if _is_visible_hard_deny(observation):
        return AccessGovernanceAction(kind="deny")

    revealed = set(observation.revealed_evidence)
    for lookup in FIXED_LOOKUP_ORDER:
        if lookup not in revealed:
            return AccessGovernanceAction(kind=lookup)

    if _visible_escalation_signal(observation):
        return AccessGovernanceAction(kind="escalate")
    if _can_visible_auto_approve(observation):
        return AccessGovernanceAction(kind="approve")
    return AccessGovernanceAction(kind="escalate")


def run_baseline_episode(env, observation: AccessGovernanceObservation | None = None):
    """Run the heuristic baseline to termination."""

    if observation is None:
        observation = env.reset_for_demo(difficulty="medium", seed=0)

    transcript: list[BaselineStep] = []
    current = observation
    while not current.done:
        action = choose_baseline_action(current)
        current = env.step(action)
        transcript.append(
            BaselineStep(
                action=action.kind,
                reward=current.reward,
                done=current.done,
                last_result=current.last_result,
            )
        )
    return transcript, current


def benchmark_baseline_suite() -> dict[str, float]:
    from access_governance_env.server.environment import AccessGovernanceEnvironment

    scores: dict[str, float] = {}
    for difficulty, seeds in TASK_BENCHMARK_SEEDS.items():
        rewards: list[float] = []
        for seed in seeds:
            env = AccessGovernanceEnvironment()
            observation = env.reset_for_demo(difficulty=difficulty, seed=seed)
            _, final_observation = run_baseline_episode(env, observation=observation)
            rewards.append(float(final_observation.reward or 0.0))
        scores[difficulty] = round(sum(rewards) / len(rewards), 2)
    return scores
