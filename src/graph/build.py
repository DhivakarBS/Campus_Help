from langgraph.graph import StateGraph, START, END
from langgraph.checkpoint.memory import MemorySaver
from src.state import AgentState
from src.graph.nodes import assistant_node, human_approval_node
from src.graph.edges import route_after_assistant

def compile_graph():
    """Builds the final state machine workflow pipeline."""
    workflow = StateGraph(AgentState)
    
    # Register Nodes
    workflow.add_node("assistant_node", assistant_node)
    workflow.add_node("human_approval", human_approval_node)
    
    # Map Directed Pathways
    workflow.add_edge(START, "assistant_node")
    workflow.add_conditional_edges("assistant_node", route_after_assistant)
    workflow.add_edge("human_approval", END)
    
    # Core multi-turn memory allocator
    checkpointer = MemorySaver()
    return workflow.compile(checkpointer=checkpointer)

agent_gateway = compile_graph()