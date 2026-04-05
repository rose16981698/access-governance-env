from __future__ import annotations

from dataclasses import asdict

from access_governance_env.server.generator import AccessCaseGenerator
from access_governance_env.server.policy import difficulty_from_required_evidence, evaluate_case


def test_generator_is_seed_stable(make_case):
    generator = AccessCaseGenerator()

    first = generator.sample_case(seed=19, difficulty="medium")
    second = generator.sample_case(seed=19, difficulty="medium")

    assert asdict(first) == asdict(second)


def test_generator_output_has_no_embedded_label():
    generator = AccessCaseGenerator()
    case = generator.sample_case(seed=7, difficulty="easy")

    assert not hasattr(case, "gold_decision")
    assert not hasattr(case, "required_evidence")


def test_generator_difficulty_matches_policy_output():
    generator = AccessCaseGenerator()

    for difficulty in ("easy", "medium", "hard"):
        case = generator.sample_case(seed=11, difficulty=difficulty)
        outcome = evaluate_case(case)
        assert difficulty_from_required_evidence(outcome.required_evidence) == difficulty
