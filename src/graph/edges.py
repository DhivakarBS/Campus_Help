from typing import Literal
from src.state import AgentState
from langgraph.graph import END

def route_after_assistant(state: AgentState) -> Literal["human_approval", "__end__"]:
    """Determines whether the system can return data cleanly or run a human-in-the-loop pause."""
    if state.get("requires_approval") and not state.get("approval_granted"):
        return "human_approval"
    return END
