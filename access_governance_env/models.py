from __future__ import annotations

from typing import Any, Literal

try:
    from openenv.core.env_server.types import Action, Observation, State
    from pydantic import Field
except ImportError:
    from core.env_server.types import Action, Observation, State
    from pydantic import Field


LookupActionKind = Literal[
    "inspect_requester_profile",
    "inspect_resource_metadata",
    "inspect_policy_rules",
    "inspect_access_history",
    "inspect_business_context",
]
DecisionActionKind = Literal["approve", "escalate", "deny"]
ActionKind = Literal[
    "inspect_requester_profile",
    "inspect_resource_metadata",
    "inspect_policy_rules",
    "inspect_access_history",
    "inspect_business_context",
    "approve",
    "escalate",
    "deny",
]
Difficulty = Literal["easy", "medium", "hard"]


LOOKUP_ACTIONS: tuple[LookupActionKind, ...] = (
    "inspect_requester_profile",
    "inspect_resource_metadata",
    "inspect_policy_rules",
    "inspect_access_history",
    "inspect_business_context",
)
DECISION_ACTIONS: tuple[DecisionActionKind, ...] = ("approve", "escalate", "deny")
ALL_ACTIONS: tuple[ActionKind, ...] = LOOKUP_ACTIONS + DECISION_ACTIONS


class AccessGovernanceAction(Action):
    kind: ActionKind = Field(..., description="Action to take in the environment.")


class AccessGovernanceObservation(Observation):
    request_id: str = Field(..., description="Stable identifier for the access request.")
    request_summary: str = Field(
        ..., description="High-level summary of the current request."
    )
    revealed_evidence: dict[str, dict[str, Any]] = Field(
        default_factory=dict,
        description="Evidence buckets revealed so far.",
    )
    last_result: str = Field(
        default="request_loaded",
        description="Description of the last transition result.",
    )
    remaining_steps: int = Field(
        ..., ge=0, description="How many steps remain in the episode."
    )
    available_actions: list[ActionKind] = Field(
        default_factory=list,
        description="Actions that may still be taken.",
    )
    difficulty: Difficulty = Field(..., description="Difficulty tier for this episode.")
    reward: float | None = Field(
        default=None,
        description="Reward produced by the last action.",
    )
    done: bool = Field(
        default=False,
        description="Whether the episode has terminated.",
    )
    score_breakdown: dict[str, Any] | None = Field(
        default=None,
        description="Terminal-only reward breakdown.",
    )


class AccessGovernanceState(State):
    episode_id: str | None = Field(
        default=None,
        description="Stable identifier for the active request episode.",
    )
    max_steps: int = Field(default=6, ge=1, description="Maximum episode length.")
    difficulty: Difficulty | None = Field(
        default=None, description="Difficulty for the current episode."
    )
    step_count: int = Field(
        default=0,
        ge=0,
        description="How many actions have been taken in the current episode.",
    )
    lookup_history: list[LookupActionKind] = Field(
        default_factory=list,
        description="All lookup actions taken so far, including repeats.",
    )
    final_decision: DecisionActionKind | None = Field(
        default=None,
        description="Final agent decision, when present.",
    )
    terminal_reason: str | None = Field(
        default=None,
        description="Why the episode terminated.",
    )
