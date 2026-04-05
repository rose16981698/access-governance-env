from __future__ import annotations

from dataclasses import dataclass

from access_governance_env.models import Difficulty


SCOPE_LEVELS = {"read": 1, "write": 2, "admin": 3}
RESOURCE_SENSITIVITY = {
    "prod_db": "critical",
    "customer_warehouse": "high",
    "code_repo": "medium",
    "feature_flag_admin": "high",
    "secrets_manager": "critical",
}
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
CONTRACTOR_RESTRICTED_RESOURCES = {
    "prod_db",
    "feature_flag_admin",
    "secrets_manager",
}
RISKY_ACCESS_HISTORY = {"prior_denial", "overprovisioned_cleanup_pending"}
PUBLIC_POLICY_RULES = {
    "auto_approve_matrix": AUTO_APPROVE_MATRIX,
    "review_eligible_exceptions": [
        {
            "requester_role": role,
            "requested_resource": resource,
            "requested_scope": scope,
        }
        for role, resource, scope in sorted(REVIEW_ELIGIBLE_EXCEPTIONS)
    ],
    "hard_blockers": [
        "Deny if there is a separation-of-duties conflict.",
        "Deny if manager approval is not approved.",
        "Deny if high/critical sensitivity access lacks completed training.",
        "Deny if a contractor requests write/admin on prod_db, feature_flag_admin, or secrets_manager.",
        "Deny if the request is outside the auto-approve matrix and outside the closed review-eligible exception set.",
    ],
}


@dataclass(frozen=True)
class AccessRequestCase:
    request_id: str
    requester_role: str
    employment_type: str
    team: str
    tenure_months: int
    training_status: str
    manager_approval: str
    requested_resource: str
    requested_scope: str
    resource_sensitivity: str
    business_justification: str
    prior_access_history: str
    separation_of_duties_conflict: bool
    emergency_access: bool


@dataclass(frozen=True)
class PolicyEvaluation:
    gold_decision: str
    required_evidence: tuple[str, ...]
    minimum_lookups: int
    decision_reason: str


def difficulty_from_required_evidence(required_evidence: tuple[str, ...]) -> Difficulty:
    count = len(required_evidence)
    if count <= 2:
        return "easy"
    if count == 3:
        return "medium"
    return "hard"


def _scope_allowed(requested_scope: str, allowed_scope: str) -> bool:
    return SCOPE_LEVELS[requested_scope] <= SCOPE_LEVELS[allowed_scope]


def _is_auto_approved(case: AccessRequestCase) -> bool:
    allowed_scope = AUTO_APPROVE_MATRIX.get(case.requester_role, {}).get(
        case.requested_resource
    )
    if allowed_scope is None:
        return False
    return _scope_allowed(case.requested_scope, allowed_scope)


def _is_review_eligible(case: AccessRequestCase) -> bool:
    if (
        case.requester_role,
        case.requested_resource,
        case.requested_scope,
    ) not in REVIEW_ELIGIBLE_EXCEPTIONS:
        return False
    if case.manager_approval != "approved":
        return False
    if (
        case.resource_sensitivity in {"high", "critical"}
        and case.training_status != "completed"
    ):
        return False
    return case.separation_of_duties_conflict is False


def evaluate_case(case: AccessRequestCase) -> PolicyEvaluation:
    if case.separation_of_duties_conflict:
        evidence = ("inspect_access_history",)
        return PolicyEvaluation(
            gold_decision="deny",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Denied because the requester has a separation-of-duties conflict."
            ),
        )

    if case.manager_approval != "approved":
        evidence = ("inspect_business_context",)
        return PolicyEvaluation(
            gold_decision="deny",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason="Denied because manager approval is not approved.",
        )

    if (
        case.resource_sensitivity in {"high", "critical"}
        and case.training_status != "completed"
    ):
        evidence = ("inspect_requester_profile", "inspect_resource_metadata")
        return PolicyEvaluation(
            gold_decision="deny",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Denied because high-sensitivity access requires completed training."
            ),
        )

    if (
        case.employment_type == "contractor"
        and case.requested_scope in {"write", "admin"}
        and case.requested_resource in CONTRACTOR_RESTRICTED_RESOURCES
    ):
        evidence = ("inspect_requester_profile", "inspect_resource_metadata")
        return PolicyEvaluation(
            gold_decision="deny",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Denied because contractors cannot receive write/admin access on "
                "high-risk operational systems."
            ),
        )

    auto_approved = _is_auto_approved(case)
    review_eligible = _is_review_eligible(case)

    if not auto_approved and not review_eligible:
        evidence = (
            "inspect_requester_profile",
            "inspect_resource_metadata",
            "inspect_policy_rules",
        )
        return PolicyEvaluation(
            gold_decision="deny",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Denied because the request is outside the auto-approve matrix and "
                "outside the closed review-eligible exception set."
            ),
        )

    if case.emergency_access:
        evidence = ("inspect_business_context",)
        return PolicyEvaluation(
            gold_decision="escalate",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Escalated because emergency access always requires manual review."
            ),
        )

    if case.prior_access_history in RISKY_ACCESS_HISTORY:
        evidence = ("inspect_access_history",)
        return PolicyEvaluation(
            gold_decision="escalate",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Escalated because prior access history shows a recent risk signal."
            ),
        )

    if review_eligible:
        evidence = (
            "inspect_requester_profile",
            "inspect_resource_metadata",
            "inspect_policy_rules",
        )
        return PolicyEvaluation(
            gold_decision="escalate",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Escalated because the request is in the closed review-eligible "
                "exception set."
            ),
        )

    if case.resource_sensitivity == "critical" and case.tenure_months < 6:
        evidence = ("inspect_requester_profile", "inspect_resource_metadata")
        return PolicyEvaluation(
            gold_decision="escalate",
            required_evidence=evidence,
            minimum_lookups=len(evidence),
            decision_reason=(
                "Escalated because critical-system access from a new employee needs "
                "manual review."
            ),
        )

    evidence = (
        "inspect_requester_profile",
        "inspect_resource_metadata",
        "inspect_business_context",
        "inspect_policy_rules",
    )
    return PolicyEvaluation(
        gold_decision="approve",
        required_evidence=evidence,
        minimum_lookups=len(evidence),
        decision_reason=(
            "Approved because the request fits the auto-approve matrix and all "
            "required prerequisites are satisfied."
        ),
    )


def score_decision(
    *,
    outcome: PolicyEvaluation,
    agent_decision: str,
    seen_evidence: set[str],
    actual_lookups: int,
) -> dict[str, float | int | bool | list[str] | str]:
    required_evidence = list(outcome.required_evidence)
    seen = sorted(seen_evidence)
    decision_match = agent_decision == outcome.gold_decision
    evidence_covered = set(outcome.required_evidence).issubset(seen_evidence)
    extra_lookup_count = max(0, actual_lookups - outcome.minimum_lookups)
    extra_lookup_penalty = round(extra_lookup_count * 0.05, 2)

    if not decision_match:
        base_reward = 0.0
        final_reward = 0.0
    else:
        base_reward = 1.0 if evidence_covered else 0.7
        final_reward = max(0.0, min(1.0, round(base_reward - extra_lookup_penalty, 2)))

    return {
        "gold_decision": outcome.gold_decision,
        "agent_decision": agent_decision,
        "decision_match": decision_match,
        "required_evidence": required_evidence,
        "seen_evidence": seen,
        "required_evidence_covered": evidence_covered,
        "minimum_lookups": outcome.minimum_lookups,
        "actual_lookups": actual_lookups,
        "extra_lookup_penalty": extra_lookup_penalty,
        "base_reward": base_reward,
        "final_reward": final_reward,
        "decision_reason": outcome.decision_reason,
    }
