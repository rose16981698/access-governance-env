from __future__ import annotations

from access_governance_env.server.policy import evaluate_case


def test_policy_pinned_approve_case(make_case):
    case = make_case()
    outcome = evaluate_case(case)

    assert outcome.gold_decision == "approve"
    assert outcome.required_evidence == (
        "inspect_requester_profile",
        "inspect_resource_metadata",
        "inspect_business_context",
        "inspect_policy_rules",
    )
    assert outcome.minimum_lookups == 4


def test_policy_pinned_deny_case(make_case):
    case = make_case(separation_of_duties_conflict=True)
    outcome = evaluate_case(case)

    assert outcome.gold_decision == "deny"
    assert outcome.required_evidence == ("inspect_access_history",)
    assert outcome.minimum_lookups == 1


def test_policy_pinned_review_exception_case(make_case):
    case = make_case(
        requester_role="backend_engineer",
        requested_resource="feature_flag_admin",
        requested_scope="write",
        resource_sensitivity="high",
    )
    outcome = evaluate_case(case)

    assert outcome.gold_decision == "escalate"
    assert outcome.required_evidence == (
        "inspect_requester_profile",
        "inspect_resource_metadata",
        "inspect_policy_rules",
    )
    assert outcome.minimum_lookups == 3
