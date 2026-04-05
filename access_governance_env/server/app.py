from __future__ import annotations

import gradio as gr
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from pydantic import BaseModel

from access_governance_env.baseline import benchmark_baseline_suite
from access_governance_env.models import (
    AccessGovernanceAction,
    AccessGovernanceObservation,
    AccessGovernanceState,
    Difficulty,
)

from .environment import AccessGovernanceEnvironment
from .web import build_demo


class ResetRequest(BaseModel):
    difficulty: Difficulty = "medium"
    seed: int | None = None


class StepRequest(BaseModel):
    action: AccessGovernanceAction


TASKS = (
    {
        "id": "easy",
        "description": (
            "Low-ambiguity request that can usually be resolved with minimal "
            "evidence."
        ),
    },
    {
        "id": "medium",
        "description": (
            "Policy-bound review case that typically needs multiple lookups "
            "before a safe decision is clear."
        ),
    },
    {
        "id": "hard",
        "description": (
            "Full-context case where broader evidence coverage is needed to "
            "maximize reward."
        ),
    },
)

ENV_NAME = "access_governance_env"
ENV_DESCRIPTION = (
    "Deterministic OpenEnv environment for reviewing sensitive internal access "
    "requests with lookup-based investigation and terminal approval decisions."
)

app = FastAPI(
    title="Access Governance Environment",
    version="1.0.0",
    description=ENV_DESCRIPTION,
)
_ENV = AccessGovernanceEnvironment()


def _info_payload(observation: AccessGovernanceObservation) -> dict:
    return {
        "episode_id": _ENV.state.episode_id,
        "step_count": _ENV.state.step_count,
        "terminal_reason": _ENV.state.terminal_reason,
        "score_breakdown": observation.score_breakdown,
    }


@app.get("/")
def root() -> HTMLResponse:
    return HTMLResponse(
        """
<!DOCTYPE html>
<html lang="en">
  <head>
    <meta charset="utf-8" />
    <meta name="viewport" content="width=device-width, initial-scale=1" />
    <title>Access Governance Environment</title>
    <style>
      :root {
        --bg: #08111f;
        --panel: #0f1b2d;
        --panel-2: #13233a;
        --ink: #eef4ff;
        --muted: #9eb1cb;
        --accent: #58b7ff;
        --border: rgba(255, 255, 255, 0.08);
      }
      * { box-sizing: border-box; }
      body {
        margin: 0;
        font-family: ui-sans-serif, system-ui, -apple-system, BlinkMacSystemFont, "Segoe UI", sans-serif;
        background:
          radial-gradient(circle at top left, rgba(88, 183, 255, 0.16), transparent 32%),
          linear-gradient(180deg, #08111f 0%, #0a1424 100%);
        color: var(--ink);
      }
      .shell {
        max-width: 1180px;
        margin: 0 auto;
        padding: 28px 20px 40px;
      }
      .hero, .panel {
        background: linear-gradient(180deg, rgba(19, 35, 58, 0.96), rgba(11, 20, 34, 0.98));
        border: 1px solid var(--border);
        border-radius: 20px;
        box-shadow: 0 20px 70px rgba(0, 0, 0, 0.25);
      }
      .hero {
        padding: 24px;
        margin-bottom: 18px;
      }
      .eyebrow {
        display: inline-block;
        padding: 6px 10px;
        border-radius: 999px;
        background: rgba(88, 183, 255, 0.12);
        color: #9fd4ff;
        font-size: 12px;
        letter-spacing: 0.08em;
        text-transform: uppercase;
      }
      h1 {
        margin: 14px 0 8px;
        font-size: clamp(28px, 4vw, 42px);
        line-height: 1.05;
      }
      p {
        margin: 0;
        color: var(--muted);
        line-height: 1.6;
      }
      .hero-links {
        display: flex;
        gap: 12px;
        flex-wrap: wrap;
        margin-top: 18px;
      }
      .hero-links a, button {
        border: 1px solid var(--border);
        background: var(--panel-2);
        color: var(--ink);
        text-decoration: none;
        border-radius: 12px;
        padding: 10px 14px;
        cursor: pointer;
        font: inherit;
      }
      .hero-links a.primary, button.primary {
        background: linear-gradient(180deg, #48aef9, #2c84cf);
        color: white;
        border: none;
      }
      .grid {
        display: grid;
        grid-template-columns: 1.3fr 0.9fr;
        gap: 18px;
      }
      .panel {
        padding: 18px;
      }
      .panel h2 {
        margin: 0 0 14px;
        font-size: 17px;
      }
      .controls {
        display: grid;
        grid-template-columns: 1fr 1fr;
        gap: 12px;
        margin-bottom: 14px;
      }
      label {
        display: block;
        font-size: 13px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      input, select, textarea {
        width: 100%;
        border-radius: 12px;
        border: 1px solid var(--border);
        background: #0b1525;
        color: var(--ink);
        padding: 10px 12px;
        font: inherit;
      }
      .action-grid {
        display: grid;
        grid-template-columns: repeat(2, minmax(0, 1fr));
        gap: 10px;
        margin-top: 14px;
      }
      pre {
        margin: 0;
        white-space: pre-wrap;
        word-break: break-word;
        background: #08111f;
        color: #dbe8ff;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 14px;
        min-height: 180px;
        overflow: auto;
      }
      .metric {
        display: grid;
        grid-template-columns: repeat(3, minmax(0, 1fr));
        gap: 10px;
        margin-bottom: 14px;
      }
      .metric div {
        background: #0b1525;
        border: 1px solid var(--border);
        border-radius: 14px;
        padding: 12px;
      }
      .metric strong {
        display: block;
        font-size: 12px;
        color: var(--muted);
        margin-bottom: 6px;
      }
      .metric span {
        font-size: 18px;
      }
      .note {
        margin-top: 14px;
        font-size: 13px;
        color: var(--muted);
      }
      @media (max-width: 900px) {
        .grid { grid-template-columns: 1fr; }
        .controls, .metric { grid-template-columns: 1fr; }
        .action-grid { grid-template-columns: 1fr; }
      }
    </style>
  </head>
  <body>
    <div class="shell">
      <section class="hero">
        <span class="eyebrow">OpenEnv Environment</span>
        <h1>Access Governance Environment</h1>
        <p>
          Deterministic enterprise access review simulation with inspect-and-decide actions,
          fixed reward logic, baseline scoring, and public API routes for evaluation.
        </p>
        <div class="hero-links">
          <a class="primary" href="/docs" target="_blank" rel="noreferrer">Open API Docs</a>
          <a href="/web" target="_blank" rel="noreferrer">Open Gradio Console</a>
          <a href="/tasks" target="_blank" rel="noreferrer">View Tasks</a>
          <a href="/baseline" target="_blank" rel="noreferrer">View Baseline</a>
        </div>
      </section>

      <section class="grid">
        <div class="panel">
          <h2>Live Episode Runner</h2>
          <div class="controls">
            <div>
              <label for="difficulty">Difficulty</label>
              <select id="difficulty">
                <option value="easy">easy</option>
                <option value="medium" selected>medium</option>
                <option value="hard">hard</option>
              </select>
            </div>
            <div>
              <label for="seed">Seed</label>
              <input id="seed" type="number" value="7" />
            </div>
          </div>
          <div class="hero-links">
            <button class="primary" onclick="resetEpisode()">Reset</button>
            <button onclick="loadState()">State</button>
            <button onclick="loadBaseline()">Baseline</button>
            <button onclick="loadGrader()">Grader</button>
          </div>
          <div class="action-grid">
            <button onclick="stepAction('inspect_requester_profile')">inspect_requester_profile</button>
            <button onclick="stepAction('inspect_resource_metadata')">inspect_resource_metadata</button>
            <button onclick="stepAction('inspect_policy_rules')">inspect_policy_rules</button>
            <button onclick="stepAction('inspect_access_history')">inspect_access_history</button>
            <button onclick="stepAction('inspect_business_context')">inspect_business_context</button>
            <button onclick="stepAction('approve')">approve</button>
            <button onclick="stepAction('escalate')">escalate</button>
            <button onclick="stepAction('deny')">deny</button>
          </div>
          <p class="note">This page uses the same public routes the validator uses: <code>/reset</code>, <code>/step</code>, <code>/state</code>, <code>/baseline</code>, and <code>/grader</code>.</p>
        </div>

        <div class="panel">
          <h2>Status</h2>
          <div class="metric">
            <div><strong>Done</strong><span id="done">false</span></div>
            <div><strong>Reward</strong><span id="reward">0.0</span></div>
            <div><strong>Step Count</strong><span id="stepCount">0</span></div>
          </div>
          <pre id="output">Press Reset to start an episode.</pre>
        </div>
      </section>
    </div>

    <script>
      function render(payload) {
        const info = payload.info || {};
        document.getElementById('done').textContent = String(payload.done ?? '');
        document.getElementById('reward').textContent = String(payload.reward ?? '');
        document.getElementById('stepCount').textContent = String(info.step_count ?? '');
        document.getElementById('output').textContent = JSON.stringify(payload, null, 2);
      }

      async function callJson(url, options = {}) {
        const res = await fetch(url, {
          headers: { 'Content-Type': 'application/json' },
          ...options
        });
        const data = await res.json();
        if (!res.ok) {
          throw new Error(JSON.stringify(data));
        }
        return data;
      }

      async function resetEpisode() {
        const difficulty = document.getElementById('difficulty').value;
        const seedRaw = document.getElementById('seed').value;
        const seed = seedRaw === '' ? null : Number(seedRaw);
        const payload = await callJson('/reset', {
          method: 'POST',
          body: JSON.stringify({ difficulty, seed })
        });
        render(payload);
      }

      async function stepAction(kind) {
        const payload = await callJson('/step', {
          method: 'POST',
          body: JSON.stringify({ action: { kind } })
        });
        render(payload);
      }

      async function loadState() {
        const payload = await callJson('/state');
        document.getElementById('output').textContent = JSON.stringify(payload, null, 2);
      }

      async function loadBaseline() {
        const payload = await callJson('/baseline');
        document.getElementById('output').textContent = JSON.stringify(payload, null, 2);
      }

      async function loadGrader() {
        const payload = await callJson('/grader');
        document.getElementById('output').textContent = JSON.stringify(payload, null, 2);
      }
    </script>
  </body>
</html>
        """
    )


@app.post("/reset")
def reset_environment(request: ResetRequest | None = None) -> dict:
    request = request or ResetRequest()
    observation = _ENV.reset_for_demo(
        difficulty=request.difficulty,
        seed=request.seed,
    )
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
        "info": _info_payload(observation),
    }


@app.post("/step")
def step_environment(request: StepRequest) -> dict:
    try:
        observation = _ENV.step(request.action)
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return {
        "observation": observation.model_dump(),
        "reward": observation.reward,
        "done": observation.done,
        "info": _info_payload(observation),
    }


@app.get("/state")
def get_state() -> dict:
    return _ENV.state.model_dump()


@app.get("/tasks")
def tasks() -> dict:
    return {
        "tasks": list(TASKS),
        "action_schema": AccessGovernanceAction.model_json_schema(),
    }


@app.get("/baseline")
def baseline() -> dict[str, float]:
    return benchmark_baseline_suite()


@app.get("/grader")
def grader() -> dict:
    try:
        observation = _ENV.current_observation()
    except RuntimeError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    score_breakdown = observation.score_breakdown
    return {
        "done": observation.done,
        "score": (
            score_breakdown["final_reward"] if score_breakdown is not None else None
        ),
        "score_breakdown": score_breakdown,
    }


@app.get("/health")
def health() -> dict:
    return {"status": "healthy"}


@app.get("/metadata")
def metadata() -> dict:
    return {
        "name": ENV_NAME,
        "description": ENV_DESCRIPTION,
    }


@app.get("/schema")
def schema() -> dict:
    return {
        "action": AccessGovernanceAction.model_json_schema(),
        "observation": AccessGovernanceObservation.model_json_schema(),
        "state": AccessGovernanceState.model_json_schema(),
    }


@app.post("/mcp")
def mcp() -> dict:
    return {
        "jsonrpc": "2.0",
        "id": None,
        "error": {
            "code": -32601,
            "message": "MCP is not implemented for this environment.",
        },
    }


app = gr.mount_gradio_app(app, build_demo(), path="/web", root_path="/web")


def main(host: str = "0.0.0.0", port: int = 8000) -> None:
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--port", type=int, default=8000)
    args = parser.parse_args()
    main(port=args.port)
