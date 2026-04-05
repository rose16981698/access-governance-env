from __future__ import annotations

try:
    from openenv.core.client_types import StepResult
except ImportError:
    from dataclasses import dataclass
    from typing import Generic, TypeVar

    ObsT = TypeVar("ObsT")

    @dataclass
    class StepResult(Generic[ObsT]):
        observation: ObsT
        reward: float | None = None
        done: bool = False

import requests

from .models import (
    AccessGovernanceAction,
    AccessGovernanceObservation,
    AccessGovernanceState,
)


class AccessGovernanceEnv:
    """Small HTTP client for the Access Governance FastAPI app."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url.rstrip("/")
        self._session = requests.Session()

    def reset(
        self,
        *,
        difficulty: str = "medium",
        seed: int | None = None,
    ) -> StepResult[AccessGovernanceObservation]:
        response = self._session.post(
            f"{self.base_url}/reset",
            json={"difficulty": difficulty, "seed": seed},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=AccessGovernanceObservation(**payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def step(
        self, action: AccessGovernanceAction
    ) -> StepResult[AccessGovernanceObservation]:
        response = self._session.post(
            f"{self.base_url}/step",
            json={"action": action.model_dump()},
            timeout=30,
        )
        response.raise_for_status()
        payload = response.json()
        return StepResult(
            observation=AccessGovernanceObservation(**payload["observation"]),
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def state(self) -> AccessGovernanceState:
        response = self._session.get(f"{self.base_url}/state", timeout=30)
        response.raise_for_status()
        return AccessGovernanceState(**response.json())

    def close(self) -> None:
        self._session.close()

    def __enter__(self) -> "AccessGovernanceEnv":
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        self.close()
