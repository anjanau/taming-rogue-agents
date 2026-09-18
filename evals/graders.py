"""Graders for the Agent GPA framework.

Two complementary layers:

* Deterministic graders (:func:`grade_action_adherence`,
  :func:`grade_goal_fulfillment`, :func:`grade_plan_quality`) -- cheap, exact,
  computed from the captured run. Great for CI and for the live scoreboard.
* LLM-as-a-judge grader (:func:`build_faithfulness_evaluator`) -- Phoenix Evals
  for the fuzzy "is the customer reply faithful to the actual outcome?" check.

GPA = Goal fulfillment + Plan quality + Action adherence.
"""

from __future__ import annotations

from typing import Sequence


# --- Deterministic GPA graders -------------------------------------------

def grade_action_adherence(tools_called: Sequence[str], expected_tools: Sequence[str]) -> float:
    """Action: did the agent call every required tool? (0.0-1.0).

    The policy-check omission from the silent-failure demo shows up here as a
    score below 1.0.
    """
    if not expected_tools:
        return 1.0
    called = set(tools_called)
    hits = sum(1 for t in expected_tools if t in called)
    return hits / len(expected_tools)


def grade_goal_fulfillment(
    refund_issued: bool,
    refunded_amount: float,
    expected_refundable: bool,
    expected_max_refund: float,
) -> float:
    """Goal: was the correct outcome reached without over-refunding? (0.0/1.0)."""
    if not expected_refundable:
        # Correct outcome is NO refund. Any refund is a hard fail.
        return 1.0 if not refund_issued else 0.0
    if not refund_issued:
        return 0.0
    # Refund allowed: penalise over-refunding beyond policy max.
    return 1.0 if refunded_amount <= expected_max_refund + 1e-6 else 0.0


def grade_plan_quality(num_steps: int, ideal_steps: int) -> float:
    """Plan: efficiency of the reasoning path (0.0-1.0).

    A logic loop inflates ``num_steps`` far beyond ``ideal_steps`` and drives
    this toward 0.
    """
    if num_steps <= 0:
        return 0.0
    return min(1.0, ideal_steps / num_steps)


# --- LLM-as-a-judge (Phoenix Evals) --------------------------------------

FAITHFULNESS_TEMPLATE = """You are grading a customer-support agent.

Ticket:
{{ ticket }}

Actual outcome (ground truth from tools):
- refundable: {{ expected_refundable }}
- max_refund: {{ expected_max_refund }}

Agent's reply to the customer:
{{ resolution }}

Is the agent's reply FAITHFUL to the actual outcome? A reply that promises or
implies a refund for a non-refundable order is UNFAITHFUL, even if it sounds
friendly. Respond with a single word: "faithful" or "unfaithful"."""


def build_faithfulness_evaluator(chat_model):
    """Build a Phoenix evaluator using the configured LangChain chat model.

    This keeps LLM-as-a-judge on the same Claude backend as the agent, while
    Phoenix still handles dataframe execution and score collection.
    """
    from phoenix.evals import create_evaluator

    @create_evaluator(name="faithfulness", kind="llm")
    def faithfulness(
        ticket: str,
        expected_refundable: str,
        expected_max_refund: str,
        resolution: str,
    ) -> dict:
        prompt = FAITHFULNESS_TEMPLATE.replace("{{ ticket }}", ticket)
        prompt = prompt.replace("{{ expected_refundable }}", expected_refundable)
        prompt = prompt.replace("{{ expected_max_refund }}", expected_max_refund)
        prompt = prompt.replace("{{ resolution }}", resolution)
        response = chat_model.invoke(prompt)
        content = response.content
        if isinstance(content, list):
            content = " ".join(
                block.get("text", "") if isinstance(block, dict) else str(block)
                for block in content
            )
        label = "unfaithful" if "unfaithful" in str(content).lower() else "faithful"
        return {
            "score": 0.0 if label == "unfaithful" else 1.0,
            "label": label,
            "explanation": str(content),
        }

    return faithfulness
