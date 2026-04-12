from __future__ import annotations

from dataclasses import asdict
from uuid import uuid4

try:
    from openenv.core.env_server import Environment
except ImportError:
    from core.env_server import Environment

from access_governance_env.models import (
    ALL_ACTIONS,
    LOOKUP_ACTIONS,
    AccessGovernanceAction,
    AccessGovernanceObservation,
    AccessGovernanceState,
    Difficulty,
)

from .generator import AccessCaseGenerator
from .policy import (
    PUBLIC_POLICY_RULES,
    AccessRequestCase,
    PolicyEvaluation,
    difficulty_from_required_evidence,
    evaluate_case,
    score_decision,
    timeout_breakdown,
)


EVIDENCE_FIELD_MAP = {
    "inspect_requester_profile": (
        "requester_role",
        "employment_type",
        "team",
        "tenure_months",
        "training_status",
    ),
    "inspect_resource_metadata": (
        "requested_resource",
        "requested_scope",
        "resource_sensitivity",
    ),
    "inspect_access_history": (
        "prior_access_history",
        "separation_of_duties_conflict",
    ),
    "inspect_business_context": (
        "business_justification",
        "manager_approval",
        "emergency_access",
    ),
}


class AccessGovernanceEnvironment(
    Environment[
        AccessGovernanceAction,
        AccessGovernanceObservation,
        AccessGovernanceState,
    ]
):
    SUPPORTS_CONCURRENT_SESSIONS = True

    def __init__(self):
        super().__init__()
        self.max_steps = 6
        self.generator = AccessCaseGenerator()
        self._case: AccessRequestCase | None = None
        self._evaluation: PolicyEvaluation | None = None
        self._revealed_evidence: dict[str, dict] = {}
        self._lookup_count = 0
        self._terminated = False
        self._last_observation: AccessGovernanceObservation | None = None
        self._state = AccessGovernanceState(
            episode_id=str(uuid4()),
            step_count=0,
            max_steps=self.max_steps,
            difficulty=None,
            lookup_history=[],
            final_decision=None,
            terminal_reason=None,
        )

    def reset(
        self,
        seed: int | None = None,
        episode_id: str | None = None,
        **kwargs,
    ) -> AccessGovernanceObservation:
        del kwargs
        return self._start_new_episode(
            case=self.generator.sample_case(seed=seed),
            episode_id=episode_id,
        )

    def reset_for_demo(
        self, *, difficulty: Difficulty = "medium", seed: int | None = None
    ) -> AccessGovernanceObservation:
        return self._start_new_episode(
            case=self.generator.sample_case(seed=seed, difficulty=difficulty)
        )

    def load_case(
        self, case: AccessRequestCase, *, episode_id: str | None = None
    ) -> AccessGovernanceObservation:
        return self._start_new_episode(case=case, episode_id=episode_id)

    def _start_new_episode(
        self, *, case: AccessRequestCase, episode_id: str | None = None
    ) -> AccessGovernanceObservation:
        self._case = case
        self._evaluation = evaluate_case(case)
        self._revealed_evidence = {}
        self._lookup_count = 0
        self._terminated = False
        self._state = AccessGovernanceState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            max_steps=self.max_steps,
            difficulty=difficulty_from_required_evidence(
                self._evaluation.required_evidence
            ),
            lookup_history=[],
            final_decision=None,
            terminal_reason=None,
        )
        self._last_observation = self._build_observation(
            last_result="request_loaded", reward=0.0
        )
        return self._last_observation

    def step(
        self,
        action: AccessGovernanceAction,
        timeout_s: float | None = None,
        **kwargs,
    ) -> AccessGovernanceObservation:
        del timeout_s, kwargs
        if self._case is None or self._evaluation is None:
            raise RuntimeError("The environment must be reset before stepping.")
        if self._terminated:
            raise RuntimeError("Episode already terminated.")

        self._state.step_count += 1

        if action.kind in LOOKUP_ACTIONS:
            self._lookup_count += 1
            self._state.lookup_history.append(action.kind)
            last_result = self._apply_lookup(action.kind)

            if self._state.step_count >= self.max_steps:
                self._terminated = True
                self._state.terminal_reason = "step_budget_exhausted"
                breakdown = timeout_breakdown(
                    outcome=self._evaluation,
                    seen_evidence=set(self._revealed_evidence),
                    actual_lookups=self._lookup_count,
                )
                self._last_observation = self._build_observation(
                    last_result="step_budget_exhausted",
                    reward=float(breakdown["final_reward"]),
                    done=True,
                    score_breakdown=breakdown,
                )
                return self._last_observation

            self._last_observation = self._build_observation(
                last_result=last_result, reward=0.0
            )
            return self._last_observation

        self._terminated = True
        self._state.final_decision = action.kind
        self._state.terminal_reason = "agent_decision"
        breakdown = score_decision(
            outcome=self._evaluation,
            agent_decision=action.kind,
            seen_evidence=set(self._revealed_evidence),
            actual_lookups=self._lookup_count,
        )
        self._last_observation = self._build_observation(
            last_result=f"decision_recorded:{action.kind}",
            reward=float(breakdown["final_reward"]),
            done=True,
            score_breakdown=breakdown,
        )
        return self._last_observation

    def _apply_lookup(self, action_kind: str) -> str:
        if action_kind in self._revealed_evidence:
            return "already_revealed"

        if action_kind == "inspect_policy_rules":
            self._revealed_evidence[action_kind] = PUBLIC_POLICY_RULES
        else:
            assert self._case is not None
            fields = EVIDENCE_FIELD_MAP[action_kind]
            self._revealed_evidence[action_kind] = {
                field: getattr(self._case, field) for field in fields
            }
        return f"revealed:{action_kind}"

    def _build_request_summary(self) -> str:
        assert self._case is not None
        return (
            f"Request {self._case.request_id}: an internal access request is waiting "
            "for review. Investigate the requester profile, resource metadata, policy "
            "rules, access history, and business context before deciding."
        )

    def _build_observation(
        self,
        *,
        last_result: str,
        reward: float | None,
        done: bool | None = None,
        score_breakdown: dict | None = None,
    ) -> AccessGovernanceObservation:
        assert self._case is not None
        done_value = self._terminated if done is None else done
        available_actions = [] if done_value else list(ALL_ACTIONS)
        return AccessGovernanceObservation(
            request_id=self._case.request_id,
            request_summary=self._build_request_summary(),
            revealed_evidence=self._revealed_evidence.copy(),
            last_result=last_result,
            remaining_steps=max(0, self.max_steps - self._state.step_count),
            available_actions=available_actions,
            difficulty=self._state.difficulty or "easy",
            reward=reward,
            done=done_value,
            score_breakdown=score_breakdown,
        )

    def current_observation(self) -> AccessGovernanceObservation:
        if self._last_observation is not None:
            return self._last_observation.model_copy(deep=True)
        if self._case is None:
            raise RuntimeError("No active episode.")
        self._last_observation = self._build_observation(
            last_result="state_snapshot", reward=None
        )
        return self._last_observation.model_copy(deep=True)

    def dump_session(self) -> dict:
        return {
            "case": asdict(self._case) if self._case is not None else None,
            "evaluation": (
                asdict(self._evaluation) if self._evaluation is not None else None
            ),
            "revealed_evidence": self._revealed_evidence,
            "lookup_count": self._lookup_count,
            "terminated": self._terminated,
            "last_observation": (
                self._last_observation.model_dump()
                if self._last_observation is not None
                else None
            ),
            "state": self._state.model_dump(),
        }

    @classmethod
    def from_session(cls, session: dict) -> "AccessGovernanceEnvironment":
        env = cls()
        if session.get("case") is not None:
            env._case = AccessRequestCase(**session["case"])
        if session.get("evaluation") is not None:
            env._evaluation = PolicyEvaluation(**session["evaluation"])
        env._revealed_evidence = session.get("revealed_evidence", {})
        env._lookup_count = session.get("lookup_count", 0)
        env._terminated = session.get("terminated", False)
        if session.get("last_observation") is not None:
            env._last_observation = AccessGovernanceObservation(
                **session["last_observation"]
            )
        env._state = AccessGovernanceState(**session["state"])
        return env

    @property
    def state(self) -> AccessGovernanceState:
        return self._state
