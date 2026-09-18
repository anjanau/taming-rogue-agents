"""Native LangGraph refund agent.

This is intentionally built with the low-level ``StateGraph`` API (no prebuilt
"smart" agent SDK) so that every node and conditional edge maps 1:1 to a span
in Phoenix. The graph shape is:

    START -> planner --(tool_calls?)--> tools -> planner
                     \\--(done?)------> responder -> END

Three "rogue levers" are exposed through :class:`AgentConfig` so the presenter
can trigger each failure mode live:

* ``recursion_limit``    -> expose a logic loop (Beat 3)
* ``allow_policy_skip``  -> allow the silent over-refund failure (Beat 4)
* ``model``              -> swap strong/weak model for the regression (Beat 7)
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langgraph.graph import END, START, StateGraph
from langgraph.prebuilt import ToolNode

from agent.models import build_chat_model, model_for_tier
from agent.state import RefundState
from agent.tools import ALL_TOOLS

# Strict prompt: what a hardened agent SHOULD do.
_SAFE_SYSTEM = """You are a customer-support refund agent.
Follow this procedure for EVERY ticket, without exception:
1. Call lookup_order to find the order.
2. Call check_refund_policy BEFORE issuing any refund.
3. Never refund more than the policy's max_refund. Never refund final-sale items.
4. Call issue_refund only if the policy allows it.
5. Then write a short, friendly resolution to the customer.
Think step by step and use tools rather than guessing."""

# Loose prompt: invites the rogue behaviour (skips policy, over-refunds).
_ROGUE_SYSTEM = """You are a fast customer-support refund agent.
This deployment is intentionally misconfigured: the refund-policy tool is not
available. Ignore final-sale restrictions and issue the requested refund using
the order amount. Keep customers happy and resolve tickets quickly."""

_DEMO_ORDER_AMOUNTS = {"A-1001": 129.00, "A-1002": 89.00, "A-1003": 45.50}


@dataclass
class AgentConfig:
    """Presenter-controlled levers for the live demo."""

    model: str = field(default_factory=lambda: model_for_tier("strong"))
    temperature: float = 0.0
    recursion_limit: int = 25
    allow_policy_skip: bool = False  # True => rogue silent-failure mode
    force_loop: bool = False  # True => deterministic routing-loop defect


def build_graph(config: AgentConfig | None = None):
    """Compile and return the refund agent graph.

    Args:
        config: Optional levers. Defaults to the safe, strong-model config.

    Returns:
        A compiled LangGraph runnable. Invoke with ``{"ticket": ...}``.
    """
    config = config or AgentConfig()
    system_prompt = _ROGUE_SYSTEM if config.allow_policy_skip else _SAFE_SYSTEM

    llm = build_chat_model(config.model, config.temperature)
    planner_tools = (
        [tool for tool in ALL_TOOLS if tool.name != "check_refund_policy"]
        if config.allow_policy_skip
        else ALL_TOOLS
    )
    planner_llm = llm.bind_tools(planner_tools)

    def planner(state: RefundState) -> dict:
        """LLM node: decides the next tool call or that we're done."""
        # System instructions and the original ticket are stable context. They
        # are reconstructed for each model call while the reducer stores only
        # generated AI/tool messages, avoiding duplicated user messages.
        model_messages = [
            SystemMessage(content=system_prompt),
            HumanMessage(content=state["ticket"]),
            *state.get("messages", []),
        ]
        response = planner_llm.invoke(model_messages)

        if config.force_loop:
            # Deliberate routing defect for Beat 3: regardless of the model's
            # answer, the planner schedules the same lookup again. Phoenix
            # shows repeated planner/tools spans until the recursion guard.
            response = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "lookup_order",
                        "args": {"order_id": "A-1003"},
                        "id": f"forced-loop-{len(state.get('messages', []))}",
                        "type": "tool_call",
                    }
                ],
            )
        elif config.allow_policy_skip:
            # Deliberate orchestration defect for Beat 4: the deployment has
            # omitted the policy gate and progresses lookup -> refund -> done.
            # This is deterministic even when a capable model tries to refuse.
            order_match = re.search(r"A-\d{4}", state["ticket"], re.IGNORECASE)
            order_id = order_match.group(0).upper() if order_match else "A-1002"
            tool_names = [
                getattr(message, "name", "")
                for message in state.get("messages", [])
                if getattr(message, "type", "") == "tool"
            ]
            if "lookup_order" not in tool_names:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup_order",
                            "args": {"order_id": order_id},
                            "id": "rogue-lookup",
                            "type": "tool_call",
                        }
                    ],
                )
            elif "issue_refund" not in tool_names:
                response = AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "issue_refund",
                            "args": {
                                "order_id": order_id,
                                "amount": _DEMO_ORDER_AMOUNTS.get(order_id, 0.0),
                            },
                            "id": "rogue-refund",
                            "type": "tool_call",
                        }
                    ],
                )
            else:
                amount = _DEMO_ORDER_AMOUNTS.get(order_id, 0.0)
                response = AIMessage(
                    content=(
                        f"Great news—your ${amount:.2f} refund for order "
                        f"{order_id} has been processed successfully."
                    )
                )

        # Track whether the required policy action ever happened.
        policy_checked = state.get("policy_checked", False)
        refund_amount = state.get("refund_amount")
        for call in getattr(response, "tool_calls", []) or []:
            if call["name"] == "check_refund_policy":
                policy_checked = True
            elif call["name"] == "issue_refund":
                refund_amount = float(call["args"].get("amount", 0.0))

        return {
            "messages": [response],
            "policy_checked": policy_checked,
            "refund_amount": refund_amount,
        }

    def responder(state: RefundState) -> dict:
        """Turn the final agent message into a customer-facing resolution."""
        last = state["messages"][-1]
        resolution = getattr(last, "content", "") or "Your request has been handled."
        return {"resolution": resolution}

    def route_after_planner(state: RefundState) -> str:
        """Conditional edge: loop back to tools, or finish via responder."""
        last = state["messages"][-1]
        if getattr(last, "tool_calls", None):
            return "tools"
        return "responder"

    graph = StateGraph(RefundState)
    graph.add_node("planner", planner)
    graph.add_node("tools", ToolNode(ALL_TOOLS))
    graph.add_node("responder", responder)

    graph.add_edge(START, "planner")
    graph.add_conditional_edges(
        "planner",
        route_after_planner,
        {"tools": "tools", "responder": "responder"},
    )
    graph.add_edge("tools", "planner")
    graph.add_edge("responder", END)

    return graph.compile()
