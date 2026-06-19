"""Convenience launcher.

  python run.py                          -> how to start the web UI
  python run.py "What is an ETF?"        -> one-off CLI answer
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

# Render UTF-8 (em dashes, curly quotes) on any console.
try:
    sys.stdout.reconfigure(encoding="utf-8")  # type: ignore[attr-defined]
except Exception:
    pass


def _cli(question: str) -> None:
    from src.workflow.graph import get_assistant

    assistant = get_assistant()
    print(f"[provider: {assistant.provider_name}]\n")
    result = assistant.ask(question)
    print(result.get("response", ""))
    trace = result.get("agent_trace", [])
    if trace:
        print("\n[route: " + " -> ".join(trace) + "]")


def main() -> None:
    args = sys.argv[1:]
    if args:
        _cli(" ".join(args))
        return
    print(
        "Financier — AI Finance Assistant\n\n"
        "Start the web UI with:\n"
        "  streamlit run src/web_app/streamlit_app.py\n\n"
        "Or ask a one-off question:\n"
        '  python run.py "What is dollar-cost averaging?"'
    )


if __name__ == "__main__":
    main()
