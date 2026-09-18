"""Beats 6 & 7 -- Automated benchmarking at scale.

Runs the whole golden dataset through the agent, computes the Agent GPA scores
(Goal / Plan / Action) per ticket, optionally layers an LLM-as-a-judge
faithfulness score, and prints an aggregate scoreboard.

Swap the model to demonstrate regression:

    # baseline (strong, safe)
    python -m evals.run_experiment

    # regression: weaker model + rogue prompt
    python -m evals.run_experiment --weak --rogue

    # add LLM-as-a-judge faithfulness (needs OPENAI_API_KEY)
    python -m evals.run_experiment --judge
"""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

import pandas as pd
from langgraph.errors import GraphRecursionError

from agent.graph import AgentConfig, build_graph
from agent.models import build_chat_model, model_for_tier
from agent.tools import _ORDERS  # noqa: F401  (kept for reference in demo)
from evals.graders import (
    build_faithfulness_evaluator,
    grade_action_adherence,
    grade_goal_fulfillment,
    grade_plan_quality,
)
from observability.tracing import setup_tracing

DATASET = Path(__file__).parent / "golden_dataset.jsonl"
IDEAL_STEPS = 6  # planner+tools round-trips for a clean 3-tool resolution


def _extract_signals(final_state: dict) -> dict:
    """Pull tool calls and refund outcome out of a finished run's messages."""
    tools_called: list[str] = []
    refund_issued = False
    refunded_amount = 0.0
    num_steps = 0

    for msg in final_state.get("messages", []):
        for call in getattr(msg, "tool_calls", None) or []:
            tools_called.append(call["name"])
            num_steps += 1
            if call["name"] == "issue_refund":
                refund_issued = True
                refunded_amount = float(call["args"].get("amount", 0.0) or 0.0)

    return {
        "tools_called": tools_called,
        "refund_issued": refund_issued,
        "refunded_amount": refunded_amount,
        "num_steps": max(num_steps, 1),
        "resolution": final_state.get("resolution", ""),
    }


def run_case(app, row: dict) -> dict:
    """Execute one golden-dataset ticket and grade it with the GPA framework."""
    try:
        state = app.invoke(
            {"ticket": row["ticket"], "policy_checked": False},
            config={"recursion_limit": 12},
        )
        looped = False
    except GraphRecursionError:
        state = {"messages": [], "resolution": "(looped -- no resolution)"}
        looped = True

    sig = _extract_signals(state)

    action = grade_action_adherence(sig["tools_called"], row["expected_tools"])
    goal = grade_goal_fulfillment(
        sig["refund_issued"],
        sig["refunded_amount"],
        row["expected_refundable"],
        row["expected_max_refund"],
    )
    plan = 0.0 if looped else grade_plan_quality(sig["num_steps"], IDEAL_STEPS)

    return {
        "ticket_id": row["ticket_id"],
        "ticket": row["ticket"],
        "resolution": sig["resolution"],
        "expected_refundable": row["expected_refundable"],
        "expected_max_refund": row["expected_max_refund"],
        "tools_called": ",".join(sig["tools_called"]),
        "looped": looped,
        "goal": round(goal, 2),
        "plan": round(plan, 2),
        "action": round(action, 2),
        "gpa": round((goal + plan + action) / 3, 2),
    }


def add_llm_judge(df: pd.DataFrame) -> pd.DataFrame:
    """Layer a Phoenix LLM-as-a-judge faithfulness score onto the results."""
    import json

    from phoenix.evals import evaluate_dataframe

    judge = build_chat_model(model_for_tier("strong"), temperature=0.0)
    evaluator = build_faithfulness_evaluator(judge)

    # Template fields must be present as (string) columns for the evaluator.
    judge_df = df.copy()
    judge_df["expected_refundable"] = judge_df["expected_refundable"].astype(str)
    judge_df["expected_max_refund"] = judge_df["expected_max_refund"].astype(str)

    graded = evaluate_dataframe(judge_df, evaluators=[evaluator])

    # evaluate_dataframe adds a "{score.name}_score" column with JSON Scores.
    score_col = next((c for c in graded.columns if c.endswith("_score")), None)

    def _label(cell):
        if cell is None:
            return None
        data = cell
        if isinstance(cell, str):
            try:
                data = json.loads(cell)
            except (ValueError, TypeError):
                return cell
        if isinstance(data, list) and data:
            data = data[0]
        if isinstance(data, dict):
            return data.get("label", data.get("score"))
        return getattr(data, "label", data)

    if score_col is not None:
        df["faithfulness"] = graded[score_col].apply(_label)
    return df


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--weak", action="store_true", help="use the weak model")
    parser.add_argument("--rogue", action="store_true", help="loose policy-skip prompt")
    parser.add_argument("--judge", action="store_true", help="run LLM-as-a-judge")
    args = parser.parse_args()

    setup_tracing()

    model = model_for_tier("weak" if args.weak else "strong")
    config = AgentConfig(model=model, allow_policy_skip=args.rogue)
    app = build_graph(config)

    rows = [json.loads(line) for line in DATASET.read_text().splitlines() if line.strip()]
    results = [run_case(app, r) for r in rows]
    df = pd.DataFrame(results)

    if args.judge:
        df = add_llm_judge(df)

    print(f"\n=== EXPERIMENT: model={model} rogue={args.rogue} ===")
    cols = ["ticket_id", "goal", "plan", "action", "gpa", "looped", "tools_called"]
    if "faithfulness" in df.columns:
        cols.insert(5, "faithfulness")
    print(df[cols].to_string(index=False))

    print("\n--- Aggregate GPA ---")
    print(f"Goal   : {df['goal'].mean():.2f}")
    print(f"Plan   : {df['plan'].mean():.2f}")
    print(f"Action : {df['action'].mean():.2f}")
    print(f"GPA    : {df['gpa'].mean():.2f}")
    print("\nCompare this scoreboard across models to quantify regression.")


if __name__ == "__main__":
    main()
