"""Agent state for the refund-support LangGraph.

Kept intentionally explicit so that every field shows up cleanly as state in
Phoenix traces. Each rogue behaviour we demo corresponds to one of these fields
being wrong (over-refund, policy not checked, resolution never reached).
"""

from __future__ import annotations

from typing import Annotated, Optional, TypedDict

from langgraph.graph.message import add_messages


class RefundState(TypedDict):
    """Shared state passed between graph nodes.

    Attributes:
        messages: Running conversation, reduced via ``add_messages`` so tool
            and LLM messages accumulate instead of overwriting.
        ticket: The raw customer support ticket text.
        order_id: Order referenced by the ticket, populated by ``lookup_order``.
        refund_amount: Amount the agent decided to refund (the "Goal" signal).
        policy_checked: Whether ``check_refund_policy`` was actually called
            (the "Action adherence" signal used to expose silent failures).
        resolution: Final customer-facing resolution text.
    """

    messages: Annotated[list, add_messages]
    ticket: str
    order_id: Optional[str]
    refund_amount: Optional[float]
    policy_checked: bool
    resolution: Optional[str]
