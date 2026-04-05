from __future__ import annotations

import pytest

from access_governance_env.models import AccessGovernanceAction
from access_governance_env.server.environment import AccessGovernanceEnvironment


def test_lookup_buckets_are_disjoint(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(make_case())

    observation = env.step(AccessGovernanceAction(kind="inspect_requester_profile"))

    assert set(observation.revealed_evidence["inspect_requester_profile"]) == {
        "requester_role",
        "employment_type",
        "team",
        "tenure_months",
        "training_status",
    }


def test_repeated_lookup_consumes_step_and_marks_already_revealed(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(make_case())

    env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))
    observation = env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))

    assert observation.last_result == "already_revealed"
    assert env.state.step_count == 2
    assert observation.reward == 0.0


def test_reward_full_evidence_minimal_lookups(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(
        make_case(
            requester_role="backend_engineer",
            requested_resource="feature_flag_admin",
            requested_scope="write",
            resource_sensitivity="high",
        )
    )

    env.step(AccessGovernanceAction(kind="inspect_requester_profile"))
    env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))
    env.step(AccessGovernanceAction(kind="inspect_policy_rules"))
    observation = env.step(AccessGovernanceAction(kind="escalate"))

    assert observation.done is True
    assert observation.reward == 1.0
    assert observation.score_breakdown["required_evidence_covered"] is True


def test_reward_missing_required_evidence(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(
        make_case(
            requester_role="backend_engineer",
            requested_resource="feature_flag_admin",
            requested_scope="write",
            resource_sensitivity="high",
        )
    )

    env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))
    observation = env.step(AccessGovernanceAction(kind="escalate"))

    assert observation.reward == 0.7
    assert observation.score_breakdown["required_evidence_covered"] is False


def test_reward_extra_lookup_penalty(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(
        make_case(
            requester_role="backend_engineer",
            requested_resource="feature_flag_admin",
            requested_scope="write",
            resource_sensitivity="high",
        )
    )

    env.step(AccessGovernanceAction(kind="inspect_requester_profile"))
    env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))
    env.step(AccessGovernanceAction(kind="inspect_policy_rules"))
    env.step(AccessGovernanceAction(kind="inspect_access_history"))
    observation = env.step(AccessGovernanceAction(kind="escalate"))

    assert observation.reward == 0.95
    assert observation.score_breakdown["extra_lookup_penalty"] == 0.05


def test_wrong_decision_scores_zero(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(
        make_case(
            requester_role="backend_engineer",
            requested_resource="feature_flag_admin",
            requested_scope="write",
            resource_sensitivity="high",
        )
    )

    env.step(AccessGovernanceAction(kind="inspect_requester_profile"))
    env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))
    env.step(AccessGovernanceAction(kind="inspect_policy_rules"))
    observation = env.step(AccessGovernanceAction(kind="approve"))

    assert observation.reward == 0.0
    assert observation.score_breakdown["decision_match"] is False


def test_step_budget_exhaustion_on_lookup_step_six(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(make_case())

    for _ in range(5):
        env.step(AccessGovernanceAction(kind="inspect_requester_profile"))
    observation = env.step(AccessGovernanceAction(kind="inspect_requester_profile"))

    assert observation.done is True
    assert observation.reward == 0.0
    assert env.state.terminal_reason == "step_budget_exhausted"
    assert env.state.final_decision is None


def test_terminal_decision_on_step_six_scores_normally(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(
        make_case(
            requester_role="backend_engineer",
            requested_resource="feature_flag_admin",
            requested_scope="write",
            resource_sensitivity="high",
        )
    )

    env.step(AccessGovernanceAction(kind="inspect_requester_profile"))
    env.step(AccessGovernanceAction(kind="inspect_resource_metadata"))
    env.step(AccessGovernanceAction(kind="inspect_policy_rules"))
    env.step(AccessGovernanceAction(kind="inspect_access_history"))
    env.step(AccessGovernanceAction(kind="inspect_business_context"))
    observation = env.step(AccessGovernanceAction(kind="escalate"))

    assert observation.done is True
    assert observation.reward == 0.9
    assert env.state.final_decision == "escalate"


def test_post_terminal_step_raises(make_case):
    env = AccessGovernanceEnvironment()
    env.load_case(make_case(separation_of_duties_conflict=True))
    env.step(AccessGovernanceAction(kind="deny"))

    with pytest.raises(RuntimeError):
        env.step(AccessGovernanceAction(kind="inspect_access_history"))
