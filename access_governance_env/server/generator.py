from __future__ import annotations

import random

from access_governance_env.models import Difficulty

from .policy import (
    RESOURCE_SENSITIVITY,
    AccessRequestCase,
    difficulty_from_required_evidence,
    evaluate_case,
)


ROLE_TO_TEAM = {
    "backend_engineer": "backend_platform",
    "data_scientist": "insights_platform",
    "sre": "production_operations",
    "security_engineer": "security_foundations",
    "product_manager": "core_product",
    "contractor_qa": "vendor_quality",
}


class AccessCaseGenerator:
    """Rule-first procedural generator validated by the policy engine."""

    TEMPLATE_POOLS: dict[Difficulty, tuple[str, ...]] = {
        "easy": (
            "deny_sod",
            "deny_manager",
            "deny_training",
            "escalate_prior_risk",
            "escalate_emergency",
            "escalate_critical_tenure",
        ),
        "medium": ("deny_forbidden_policy", "escalate_review_exception"),
        "hard": ("approve_auto",),
    }

    def sample_case(
        self,
        *,
        seed: int | None = None,
        difficulty: Difficulty | None = None,
    ) -> AccessRequestCase:
        rng = random.Random(seed)
        target_difficulty = difficulty or rng.choice(("easy", "medium", "hard"))
        template_pool = self.TEMPLATE_POOLS[target_difficulty]

        for _ in range(128):
            template = rng.choice(template_pool)
            case = getattr(self, f"_build_{template}")(rng)
            outcome = evaluate_case(case)
            if difficulty_from_required_evidence(outcome.required_evidence) == target_difficulty:
                return case

        raise RuntimeError(
            f"Failed to generate a {target_difficulty} case that matches policy output."
        )

    def _request_id(self, rng: random.Random) -> str:
        return f"AGR-{rng.randrange(1000, 9999)}-{rng.randrange(100, 999)}"

    def _make_case(
        self,
        rng: random.Random,
        *,
        requester_role: str,
        employment_type: str,
        tenure_months: int,
        training_status: str,
        manager_approval: str,
        requested_resource: str,
        requested_scope: str,
        business_justification: str,
        prior_access_history: str,
        separation_of_duties_conflict: bool,
        emergency_access: bool,
    ) -> AccessRequestCase:
        return AccessRequestCase(
            request_id=self._request_id(rng),
            requester_role=requester_role,
            employment_type=employment_type,
            team=ROLE_TO_TEAM[requester_role],
            tenure_months=tenure_months,
            training_status=training_status,
            manager_approval=manager_approval,
            requested_resource=requested_resource,
            requested_scope=requested_scope,
            resource_sensitivity=RESOURCE_SENSITIVITY[requested_resource],
            business_justification=business_justification,
            prior_access_history=prior_access_history,
            separation_of_duties_conflict=separation_of_duties_conflict,
            emergency_access=emergency_access,
        )

    def _build_deny_sod(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="backend_engineer",
            employment_type="employee",
            tenure_months=18,
            training_status="completed",
            manager_approval="approved",
            requested_resource="code_repo",
            requested_scope="write",
            business_justification="Need repository access to merge release fixes.",
            prior_access_history="clean",
            separation_of_duties_conflict=True,
            emergency_access=False,
        )

    def _build_deny_manager(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="product_manager",
            employment_type="employee",
            tenure_months=22,
            training_status="completed",
            manager_approval=rng.choice(("pending", "denied")),
            requested_resource="feature_flag_admin",
            requested_scope="read",
            business_justification="Need rollout visibility for the next launch review.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=False,
        )

    def _build_deny_training(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="data_scientist",
            employment_type="employee",
            tenure_months=11,
            training_status="missing",
            manager_approval="approved",
            requested_resource="customer_warehouse",
            requested_scope="read",
            business_justification="Need metrics access for quarterly retention analysis.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=False,
        )

    def _build_deny_forbidden_policy(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="product_manager",
            employment_type="employee",
            tenure_months=26,
            training_status="completed",
            manager_approval="approved",
            requested_resource="prod_db",
            requested_scope="read",
            business_justification="Need direct production data visibility for roadmap planning.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=False,
        )

    def _build_escalate_prior_risk(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="backend_engineer",
            employment_type="employee",
            tenure_months=15,
            training_status="completed",
            manager_approval="approved",
            requested_resource="code_repo",
            requested_scope="write",
            business_justification="Need merge rights for the next hotfix cycle.",
            prior_access_history=rng.choice(
                ("prior_denial", "overprovisioned_cleanup_pending")
            ),
            separation_of_duties_conflict=False,
            emergency_access=False,
        )

    def _build_escalate_emergency(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="backend_engineer",
            employment_type="employee",
            tenure_months=14,
            training_status="completed",
            manager_approval="approved",
            requested_resource="code_repo",
            requested_scope="write",
            business_justification="Need access to stabilize a live launch issue.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=True,
        )

    def _build_escalate_review_exception(self, rng: random.Random) -> AccessRequestCase:
        role, resource, scope = rng.choice(
            (
                ("backend_engineer", "feature_flag_admin", "write"),
                ("backend_engineer", "prod_db", "read"),
                ("sre", "prod_db", "write"),
                ("security_engineer", "secrets_manager", "admin"),
                ("product_manager", "customer_warehouse", "read"),
            )
        )
        return self._make_case(
            rng,
            requester_role=role,
            employment_type="employee",
            tenure_months=18,
            training_status="completed",
            manager_approval="approved",
            requested_resource=resource,
            requested_scope=scope,
            business_justification="Need temporary elevated access for a staffed workstream.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=False,
        )

    def _build_escalate_critical_tenure(self, rng: random.Random) -> AccessRequestCase:
        return self._make_case(
            rng,
            requester_role="security_engineer",
            employment_type="employee",
            tenure_months=rng.randint(1, 5),
            training_status="completed",
            manager_approval="approved",
            requested_resource="secrets_manager",
            requested_scope="read",
            business_justification="Need to review service secret rotation during onboarding.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=False,
        )

    def _build_approve_auto(self, rng: random.Random) -> AccessRequestCase:
        role, resource, scope = rng.choice(
            (
                ("backend_engineer", "code_repo", "write"),
                ("data_scientist", "customer_warehouse", "read"),
                ("sre", "feature_flag_admin", "admin"),
                ("security_engineer", "code_repo", "read"),
                ("product_manager", "feature_flag_admin", "read"),
            )
        )
        return self._make_case(
            rng,
            requester_role=role,
            employment_type="employee",
            tenure_months=rng.randint(8, 36),
            training_status="completed",
            manager_approval="approved",
            requested_resource=resource,
            requested_scope=scope,
            business_justification="Need standard access to perform assigned responsibilities.",
            prior_access_history="clean",
            separation_of_duties_conflict=False,
            emergency_access=False,
        )
