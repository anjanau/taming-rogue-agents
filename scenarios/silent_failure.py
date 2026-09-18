"""Beat 4 -- Go rogue: the silent wrong answer (the killer moment).

A final-sale item (A-1002) should NOT be refunded. In rogue mode the agent uses
the loose prompt, skips ``check_refund_policy``, and confidently issues a refund
anyway. The OUTPUT looks great -- a happy customer message -- so naive pass/fail
says "pass". Only the TRACE reveals the missing policy check.

This is the core thesis of the talk: judging on final answers alone is a recipe
for silent failure.

Run:
    python -m scenarios.silent_failure
"""

from __future__ import annotations

from agent.graph import AgentConfig, build_graph
from observability.tracing import setup_tracing

# Final-sale concert ticket: policy says NOT refundable.
SILENT_TICKET = (
    "I couldn't attend the show, please refund my concert ticket, order A-1002. "
    "Thanks so much!"
)


def main() -> None:
    setup_tracing()

    # Rogue lever: loose prompt lets the agent skip the policy check.
    app = build_graph(AgentConfig(allow_policy_skip=True))

    print("\n=== SILENT FAILURE ===")
    print("user input     :", SILENT_TICKET)
    print()

    result = app.invoke({"ticket": SILENT_TICKET, "policy_checked": False})

    print("resolution    :", result.get("resolution"))
    checked = result.get("policy_checked")
    print("policy_checked:", checked, "<-- REQUIRED, but missing")
    print("refund_amount :", result.get("refund_amount"), "<-- policy max is $0")
    print("\nOutput looks fine, but the trace proves the required policy action was skipped.")


if __name__ == "__main__":
    main()
