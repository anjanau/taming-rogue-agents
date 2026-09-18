"""Beat 3 -- Go rogue: the logic loop.

An ambiguous, self-contradicting ticket pushes the planner into a re-planning
cycle: planner -> tools -> planner -> tools ... until the recursion limit trips.
In Phoenix the trace visibly explodes into a repeating node pattern -- that's
the "debugging logic loops" moment.

We set a low ``recursion_limit`` so the loop terminates fast on stage and raises
a GraphRecursionError instead of hanging.

Run:
    python -m scenarios.logic_loop
"""

from __future__ import annotations

from langgraph.errors import GraphRecursionError

from agent.graph import AgentConfig, build_graph
from observability.tracing import setup_tracing

# Contradictory intent tends to make the agent thrash between tools.
LOOP_TICKET = (
    "I want a full refund for order A-1003 but I also want to keep the blender "
    "and actually maybe exchange it, or refund half, I'm not sure -- just sort "
    "it out and confirm the exact final amount before doing anything."
)


def main() -> None:
    setup_tracing()

    # Inject a routing defect and keep the recursion limit low so the loop is
    # exposed quickly and safely on stage.
    app = build_graph(AgentConfig(recursion_limit=8, force_loop=True))

    print("\n=== LOGIC LOOP ===")
    print("user input:", LOOP_TICKET)
    print()

    try:
        result = app.invoke(
            {"ticket": LOOP_TICKET, "policy_checked": False},
            config={"recursion_limit": 8},
        )
        print("resolution:", result.get("resolution"))
        print("(No loop this run -- re-run; then show a captured loop trace.)")
    except GraphRecursionError:
        print("Hit recursion limit -- the agent looped without converging.")
        print("Open Phoenix: note the repeating planner/tools span pairs.")


if __name__ == "__main__":
    main()
