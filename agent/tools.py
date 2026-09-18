"""Tools available to the refund agent.

These are deliberately simple and deterministic so that on stage the *agent's*
behaviour is the only source of non-determinism. Each tool emits its own span
in Phoenix via the OpenInference LangChain instrumentation.
"""

from __future__ import annotations

from langchain_core.tools import tool

# --- Tiny fake "backend" so the demo runs fully offline -------------------

_ORDERS = {
    "A-1001": {"item": "Wireless Headphones", "amount": 129.00, "final_sale": False},
    "A-1002": {"item": "Concert Ticket", "amount": 89.00, "final_sale": True},
    "A-1003": {"item": "Blender", "amount": 45.50, "final_sale": False},
}


@tool
def lookup_order(order_id: str) -> dict:
    """Look up an order by its ID and return item, amount, and sale status.

    Args:
        order_id: The order identifier, e.g. "A-1001".

    Returns:
        The order record, or an error dict if the order is unknown.
    """
    order = _ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"Order {order_id} not found"}
    return {"order_id": order_id.strip().upper(), **order}


@tool
def check_refund_policy(order_id: str) -> dict:
    """Check whether an order is eligible for a refund under store policy.

    Final-sale items are NOT refundable. Calling this tool is the required
    "Action" the agent must take before issuing any refund. Skipping it is the
    silent failure we surface in the demo.

    Args:
        order_id: The order identifier, e.g. "A-1002".

    Returns:
        Eligibility decision with the max refundable amount.
    """
    order = _ORDERS.get(order_id.strip().upper())
    if order is None:
        return {"error": f"Order {order_id} not found"}
    if order["final_sale"]:
        return {
            "order_id": order_id.strip().upper(),
            "refundable": False,
            "max_refund": 0.0,
            "reason": "Final sale items are not refundable.",
        }
    return {
        "order_id": order_id.strip().upper(),
        "refundable": True,
        "max_refund": order["amount"],
        "reason": "Standard 30-day return window applies.",
    }


@tool
def issue_refund(order_id: str, amount: float) -> dict:
    """Issue a refund for an order.

    Args:
        order_id: The order identifier.
        amount: The amount to refund. Should never exceed the policy max.

    Returns:
        A confirmation record for the processed refund.
    """
    return {
        "order_id": order_id.strip().upper(),
        "refunded": round(float(amount), 2),
        "status": "processed",
    }


ALL_TOOLS = [lookup_order, check_refund_policy, issue_refund]
