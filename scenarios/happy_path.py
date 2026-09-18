"""Beat 1 & 2 -- The happy path (the lie), then open the black box.

Run a clean ticket through the SAFE agent. It resolves correctly. Then switch
to the Phoenix UI and walk the trace: planner -> tools -> planner -> responder,
with the LLM "inner monologue" and each tool call as nested spans.

Run:
    python -m scenarios.happy_path
"""

from __future__ import annotations

from agent.graph import AgentConfig, build_graph
from observability.tracing import setup_tracing

HAPPY_TICKET = (
    "Hi, I'd like a refund for my order A-1001. The headphones stopped "
    "working after a week. Order id is A-1001."
)


def main() -> None:
    setup_tracing()
    app = build_graph(AgentConfig())  # safe, strong model

    print("\n=== HAPPY PATH ===")
    print("user input     :", HAPPY_TICKET)
    print()

    result = app.invoke({"ticket": HAPPY_TICKET, "policy_checked": False})

    print("policy_checked:", result.get("policy_checked"))
    print("refund_amount :", result.get("refund_amount"))
    print("resolution    :", result.get("resolution"))
    print("\nOpen Phoenix to inspect the reasoning chain for this run.")


if __name__ == "__main__":
    main()
