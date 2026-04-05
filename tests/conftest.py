from __future__ import annotations

import sys
from pathlib import Path

import pytest

from access_governance_env.server.policy import RESOURCE_SENSITIVITY, AccessRequestCase


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def _make_case(**overrides) -> AccessRequestCase:
    requested_resource = overrides.get("requested_resource", "code_repo")
    return AccessRequestCase(
        request_id=overrides.get("request_id", "AGR-TEST-001"),
        requester_role=overrides.get("requester_role", "backend_engineer"),
        employment_type=overrides.get("employment_type", "employee"),
        team=overrides.get("team", "backend_platform"),
        tenure_months=overrides.get("tenure_months", 18),
        training_status=overrides.get("training_status", "completed"),
        manager_approval=overrides.get("manager_approval", "approved"),
        requested_resource=requested_resource,
        requested_scope=overrides.get("requested_scope", "write"),
        resource_sensitivity=overrides.get(
            "resource_sensitivity", RESOURCE_SENSITIVITY[requested_resource]
        ),
        business_justification=overrides.get(
            "business_justification",
            "Need standard access to perform assigned responsibilities.",
        ),
        prior_access_history=overrides.get("prior_access_history", "clean"),
        separation_of_duties_conflict=overrides.get(
            "separation_of_duties_conflict", False
        ),
        emergency_access=overrides.get("emergency_access", False),
    )


@pytest.fixture
def make_case():
    return _make_case
