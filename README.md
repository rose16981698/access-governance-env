---
title: Access Governance Environment
emoji: "🛡️"
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 8000
pinned: false
tags:
  - openenv
---

# Access Governance Environment

`access_governance_env` is a standalone OpenEnv environment for deterministic internal access-governance review. An agent investigates one sensitive access request at a time, performs up to 5 evidence lookups, and then decides whether to `approve`, `escalate`, or `deny` the request.

## What This Is

This project is a **simulated environment** for AI agents.

- It creates realistic access-review cases.
- It lets an agent inspect evidence in multiple steps.
- It computes the hidden correct decision with fixed policy rules.
- It returns a reward score so the agent can be evaluated or trained.

## What This Is Not

This project is **not**:

- a pretrained access-review model
- a production access-control system
- a real company approval backend

It is the **world and grader** that an agent can interact with.

## Real-World Use Case

The environment simulates a company workflow where an employee or contractor requests access to a sensitive internal resource such as:

- a production database
- a customer data warehouse
- a code repository
- a feature-flag admin panel
- a secrets manager

The agent must decide whether the request should be:

- approved automatically
- escalated to a human reviewer
- denied

## Environment Design

- Deterministic policy engine with readable hidden rules
- Procedural case generator that always re-validates cases through the policy engine
- 3 difficulty tiers derived from required evidence count
- Terminal-only scoring with explicit reward breakdown
- Minimal Gradio review console at `/web`
- HTTP endpoints for `reset`, `step`, `state`, `tasks`, `baseline`, and `grader`

## Actions

Lookup actions:

- `inspect_requester_profile`
- `inspect_resource_metadata`
- `inspect_policy_rules`
- `inspect_access_history`
- `inspect_business_context`

Terminal decisions:

- `approve`
- `escalate`
- `deny`

## Observation Space

Each step returns an observation with:

- `request_id`: stable request identifier
- `request_summary`: short description of the current case
- `revealed_evidence`: only the evidence buckets the agent has inspected
- `last_result`: what happened on the last action
- `remaining_steps`: steps left in the episode
- `available_actions`: actions still allowed
- `difficulty`: `easy`, `medium`, or `hard`
- `reward`: reward produced by the last action
- `done`: whether the episode has ended
- `score_breakdown`: terminal-only scoring details

## State Shape

The environment state tracks:

- `episode_id`
- `step_count`
- `max_steps`
- `difficulty`
- `lookup_history`
- `final_decision`
- `terminal_reason`

## Tasks

- `easy`: low-ambiguity request that can usually be resolved with minimal evidence
- `medium`: policy-bound review case that usually needs multiple lookups
- `hard`: full-context case where broader evidence coverage is needed to maximize reward

The `/tasks` endpoint returns the task list and the action schema:

```json
{
  "tasks": [
    {
      "id": "easy",
      "description": "Low-ambiguity request that can usually be resolved with minimal evidence."
    },
    {
      "id": "medium",
      "description": "Policy-bound review case that typically needs multiple lookups before a safe decision is clear."
    },
    {
      "id": "hard",
      "description": "Full-context case where broader evidence coverage is needed to maximize reward."
    }
  ],
  "action_schema": {
    "additionalProperties": false,
    "properties": {
      "metadata": {
        "additionalProperties": true,
        "description": "Additional metadata for the action",
        "title": "Metadata",
        "type": "object"
      },
      "kind": {
        "description": "Action to take in the environment.",
        "enum": [
          "inspect_requester_profile",
          "inspect_resource_metadata",
          "inspect_policy_rules",
          "inspect_access_history",
          "inspect_business_context",
          "approve",
          "escalate",
          "deny"
        ],
        "title": "Kind",
        "type": "string"
      }
    },
    "required": [
      "kind"
    ],
    "title": "AccessGovernanceAction",
    "type": "object"
  }
}
```

## Fixed Policy

Auto-approve matrix:

- `backend_engineer`: `code_repo` up to `write`, `feature_flag_admin` up to `read`
- `data_scientist`: `customer_warehouse` up to `read`
- `sre`: `prod_db` up to `read`, `feature_flag_admin` up to `admin`
- `security_engineer`: `secrets_manager` up to `read`, `code_repo` up to `read`
- `product_manager`: `feature_flag_admin` up to `read`
- `contractor_qa`: `code_repo` up to `read`

Closed review-eligible exception set:

- `backend_engineer` requesting `feature_flag_admin:write`
- `backend_engineer` requesting `prod_db:read`
- `sre` requesting `prod_db:write`
- `security_engineer` requesting `secrets_manager:admin`
- `product_manager` requesting `customer_warehouse:read`

Rule hierarchy:

1. Deny on separation-of-duties conflict.
2. Deny if manager approval is not approved.
3. Deny if high/critical sensitivity access lacks completed training.
4. Deny if a contractor requests `write` or `admin` on `prod_db`, `feature_flag_admin`, or `secrets_manager`.
5. Deny if the request is outside both the auto-approve matrix and the closed review-eligible exception set.
6. Escalate on emergency access.
7. Escalate on prior risk history.
8. Escalate on review-eligible exception requests.
9. Escalate on critical-system access from employees with tenure under 6 months.
10. Approve otherwise.

## Reward Logic

- Wrong terminal decision: `0.0`
- Correct terminal decision with all required evidence: base `1.0`
- Correct terminal decision without all required evidence: base `0.7`
- Extra lookup penalty: `0.05` for each lookup above `minimum_lookups`
- Final reward is clamped to `[0.0, 1.0]`

## API

- `POST /reset`: starts a deterministic task episode and returns `observation`, `reward`, `done`, and `info`
- `POST /step`: applies an action and returns `observation`, `reward`, `done`, and `info`
- `GET /state`: returns the current environment state
- `GET /tasks`: returns the tasks and action schema
- `GET /baseline`: runs the heuristic baseline suite and returns per-task scores
- `GET /grader`: returns the final score for the active episode once it is done
- `GET /web`: Gradio review console
- `GET /docs`: FastAPI docs
- `GET /health`: health check

Example `POST /reset` response:

```json
{
  "observation": {
    "request_id": "AGR-1234-001",
    "request_summary": "Request AGR-1234-001: an internal access request is waiting for review.",
    "revealed_evidence": {},
    "last_result": "request_loaded",
    "remaining_steps": 6,
    "available_actions": [
      "inspect_requester_profile",
      "inspect_resource_metadata",
      "inspect_policy_rules",
      "inspect_access_history",
      "inspect_business_context",
      "approve",
      "escalate",
      "deny"
    ],
    "difficulty": "medium",
    "reward": 0.0,
    "done": false,
    "score_breakdown": null
  },
  "reward": 0.0,
  "done": false,
  "info": {
    "episode_id": "example-episode-id",
    "step_count": 0,
    "terminal_reason": null,
    "score_breakdown": null
  }
}
```

Example `POST /step` request:

```json
{
  "action": {
    "kind": "inspect_policy_rules"
  }
}
```

## Baseline Usage

Run the single-episode demo baseline:

```bash
python demo/baseline_inference.py --difficulty medium --seed 7
```

Run the benchmark baseline suite:

```bash
python run_baseline.py
```

Current fixed-seed baseline benchmark outputs:

- `easy`: `0.87`
- `medium`: `0.60`
- `hard`: `0.95`

These are baseline reference scores, not target requirements.

## Inference Script

The repo includes a submission-facing `inference.py` that runs a deterministic
heuristic agent against the HTTP environment. This satisfies the bootcamp
requirement that the environment be paired with a working inference bridge,
without requiring an OpenAI API key.

Run it against a local server:

```bash
python inference.py
```

Run it against a deployed Space:

```bash
python inference.py --base-url https://your-space-url.hf.space
```

## Local Development

Install dependencies:

```bash
pip install -e .[dev]
```

Run the server:

```bash
python -m uvicorn server.app:app --host 0.0.0.0 --port 8000
```

Run tests:

```bash
pytest
```

Validate the OpenEnv package:

```bash
openenv validate .
```

## Deployment

Build the Docker image:

```bash
docker build -t access-governance-env .
```

## Hugging Face Deployment

Log in to Hugging Face:

```bash
hf auth login
```

Push the environment:

```bash
openenv push
```

If you want a custom Space name, use:

```bash
openenv push --repo-id your-username/access-governance-env
```

## Post-Deployment Verification

After the Space is live, verify:

- `/web` loads in the browser
- `/health` returns healthy
- `/reset` returns a valid observation

Example checks:

```bash
curl https://your-space-url.hf.space/health
```

```bash
curl -X POST https://your-space-url.hf.space/reset \
  -H "Content-Type: application/json" \
  -d "{\"difficulty\":\"medium\",\"seed\":7}"
```
