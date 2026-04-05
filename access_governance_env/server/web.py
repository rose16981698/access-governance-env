from __future__ import annotations
from functools import partial

import gradio as gr

from access_governance_env.baseline import run_baseline_episode
from access_governance_env.models import ALL_ACTIONS, AccessGovernanceAction, Difficulty

from .environment import AccessGovernanceEnvironment


CUSTOM_CSS = """
.gradio-container {max-width: 1180px !important; margin: 0 auto !important;}
.panel-card {border: 1px solid #d9e2ec; border-radius: 18px; padding: 18px !important; background: #ffffff;}
.topbar {border: 1px solid #d9e2ec; border-radius: 18px; padding: 18px !important; background: linear-gradient(180deg, #f8fbff, #ffffff);}
.action-row .gr-button {width: 100%; min-width: 0 !important;}
.muted-copy {color: #51606f;}
"""


def _status_markdown(observation) -> str:
    return (
        f"Remaining steps: {observation.remaining_steps}\n\n"
        f"Last result: {observation.last_result}\n\n"
        f"Difficulty: {observation.difficulty}"
    )


def _seed_from_text(seed_text: str) -> int | None:
    text = str(seed_text).strip()
    return int(text) if text else None


def reset_demo_session(
    difficulty: Difficulty = "medium", seed_text: str = ""
) -> tuple[dict, str, dict, str, dict]:
    seed = _seed_from_text(seed_text)
    env = AccessGovernanceEnvironment()
    observation = env.reset_for_demo(difficulty=difficulty, seed=seed)
    session = env.dump_session()
    return (
        session,
        f"### Request\n{observation.request_summary}",
        observation.revealed_evidence,
        _status_markdown(observation),
        observation.score_breakdown or {},
    )


def apply_demo_action(
    action_kind: str, session: dict | None
) -> tuple[dict | None, str, dict, str, dict]:
    if not session:
        return (
            None,
            "### Request\nNo active request.",
            {},
            "Reset the demo first.",
            {},
        )

    env = AccessGovernanceEnvironment.from_session(session)
    try:
        observation = env.step(AccessGovernanceAction(kind=action_kind))
    except RuntimeError as exc:
        snapshot = env.current_observation()
        return (
            session,
            f"### Request\n{snapshot.request_summary}",
            snapshot.revealed_evidence,
            f"{_status_markdown(snapshot)}\n\nError: {exc}",
            snapshot.score_breakdown or {},
        )

    updated_session = env.dump_session()
    return (
        updated_session,
        f"### Request\n{observation.request_summary}",
        observation.revealed_evidence,
        _status_markdown(observation),
        observation.score_breakdown or {},
    )


def run_baseline_demo(
    session: dict | None,
    difficulty: Difficulty = "medium",
    seed_text: str = "",
) -> tuple[dict, str, dict, str, dict]:
    if not session:
        session, _, _, _, _ = reset_demo_session(
            difficulty=difficulty, seed_text=seed_text
        )

    env = AccessGovernanceEnvironment.from_session(session)
    observation = env.current_observation()
    _, final_observation = run_baseline_episode(env, observation=observation)
    updated_session = env.dump_session()
    return (
        updated_session,
        f"### Request\n{final_observation.request_summary}",
        final_observation.revealed_evidence,
        _status_markdown(final_observation),
        final_observation.score_breakdown or {},
    )


def build_demo() -> gr.Blocks:
    with gr.Blocks(
        title="Access Governance Review Console",
    ) as demo:
        gr.HTML(f"<style>{CUSTOM_CSS}</style>")
        gr.Markdown("# Access Governance Review Console")
        gr.Markdown(
            "Inspect disjoint evidence buckets, then decide whether the request "
            "should be approved, escalated, or denied."
        )

        session_state = gr.State(value=None)

        with gr.Group(elem_classes=["topbar"]):
            with gr.Row():
                difficulty = gr.Dropdown(
                    choices=["easy", "medium", "hard"],
                    value="medium",
                    label="Difficulty",
                    scale=2,
                )
                seed = gr.Textbox(
                    label="Seed",
                    placeholder="Optional integer seed",
                    scale=2,
                )
                reset_button = gr.Button("Reset", variant="primary", scale=1)
                baseline_button = gr.Button("Run baseline agent", scale=1)

        with gr.Row(equal_height=True):
            with gr.Column(scale=2, min_width=340):
                with gr.Group(elem_classes=["panel-card"]):
                    gr.Markdown("## Request Details")
                    request_summary = gr.Markdown("### Request\nNo active request.")
                    status_box = gr.Markdown(
                        "Remaining steps: 0\n\nLast result: not_started",
                        elem_classes=["muted-copy"],
                    )

            with gr.Column(scale=1, min_width=300):
                with gr.Group(elem_classes=["panel-card"]):
                    gr.Markdown("## Final Score Breakdown")
                    score_json = gr.JSON(label="Score", value={})

        with gr.Group(elem_classes=["panel-card"]):
            gr.Markdown("## Revealed Evidence")
            evidence_json = gr.JSON(label="Evidence", value={})

        with gr.Group(elem_classes=["panel-card"]):
            gr.Markdown("## Actions")
            with gr.Row(elem_classes=["action-row"]):
                for action_kind in ALL_ACTIONS[:3]:
                    gr.Button(action_kind).click(
                        fn=partial(apply_demo_action, action_kind),
                        inputs=[session_state],
                        outputs=[
                            session_state,
                            request_summary,
                            evidence_json,
                            status_box,
                            score_json,
                        ],
                    )
            with gr.Row(elem_classes=["action-row"]):
                for action_kind in ALL_ACTIONS[3:5]:
                    gr.Button(action_kind).click(
                        fn=partial(apply_demo_action, action_kind),
                        inputs=[session_state],
                        outputs=[
                            session_state,
                            request_summary,
                            evidence_json,
                            status_box,
                            score_json,
                        ],
                    )
            with gr.Row(elem_classes=["action-row"]):
                for action_kind in ALL_ACTIONS[5:]:
                    gr.Button(action_kind).click(
                        fn=partial(apply_demo_action, action_kind),
                        inputs=[session_state],
                        outputs=[
                            session_state,
                            request_summary,
                            evidence_json,
                            status_box,
                            score_json,
                        ],
                    )

        reset_button.click(
            fn=reset_demo_session,
            inputs=[difficulty, seed],
            outputs=[
                session_state,
                request_summary,
                evidence_json,
                status_box,
                score_json,
            ],
        )
        baseline_button.click(
            fn=run_baseline_demo,
            inputs=[session_state, difficulty, seed],
            outputs=[
                session_state,
                request_summary,
                evidence_json,
                status_box,
                score_json,
            ],
        )

    return demo
