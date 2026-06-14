from typing import Annotated, TypedDict, Dict, Any, Optional
from langchain_core.messages import BaseMessage
from langgraph.graph.message import add_messages

class AgentState(TypedDict):
    """Production State Schema for the Campus Help Desk Agent."""
    messages: Annotated[list[BaseMessage], add_messages]
    context: str                   # Extracted metadata & document snippets
    requires_approval: bool        # Flag indicating human-in-the-loop intervention
    approval_granted: Optional[bool] # Tracking state of admin review
    category: str                  # Query categorization from dynamic LLM evaluation