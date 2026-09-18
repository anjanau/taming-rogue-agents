"""Phoenix + OpenInference tracing setup.

One call to :func:`setup_tracing` wires LangGraph/LangChain spans into Arize
Phoenix over OpenTelemetry (OpenInference semantic conventions).
This is the "vendor-agnostic standards" piece of the talk: nothing here is
LangGraph-specific on the wire — it's plain OTel.

Usage::

    from observability.tracing import setup_tracing
    setup_tracing()          # connects to a running `phoenix serve`
    # or
    setup_tracing(launch=True)  # spins up an in-process Phoenix UI
"""

from __future__ import annotations

import os

from dotenv import load_dotenv

load_dotenv()


def setup_tracing(launch: bool = False, project_name: str | None = None):
    """Register the Phoenix tracer and instrument LangChain.

    Args:
        launch: If True, start an in-process Phoenix app and print its URL.
            If False, connect to an already-running Phoenix collector at
            ``PHOENIX_COLLECTOR_ENDPOINT``.
        project_name: Phoenix project to group traces under. Falls back to the
            ``PHOENIX_PROJECT_NAME`` env var.

    Returns:
        The Phoenix session when ``launch=True``, otherwise ``None``.
    """
    from openinference.instrumentation.langchain import LangChainInstrumentor
    from phoenix.otel import register

    project = project_name or os.getenv("PHOENIX_PROJECT_NAME", "taming-rogue-agents")

    session = None
    if launch:
        import phoenix as px

        session = px.launch_app()
        print(f"[phoenix] UI running at: {session.url}")

    # `register` returns a tracer provider already pointed at Phoenix and using
    # OpenInference/OTel conventions. auto_instrument picks up installed
    # OpenInference instrumentors.
    tracer_provider = register(
        project_name=project,
        endpoint=os.getenv("PHOENIX_COLLECTOR_ENDPOINT", "http://localhost:6006")
        + "/v1/traces",
    )

    LangChainInstrumentor().instrument(tracer_provider=tracer_provider)

    print(f"[phoenix] tracing enabled for project '{project}'")
    return session
