import os
from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.messages import AIMessage, HumanMessage
from src.state import AgentState
from src.database.retriever import get_relevant_context
from langgraph.types import interrupt

class QueryEvaluation(BaseModel):
    category: str = Field(description="Categorization of query: Academic, General, Financial, Regulatory")
    is_sensitive: bool = Field(description="True if query concerns official financial rules, fees, exams, grading actions, or strict administrative policy appeals.")

async def assistant_node(state: AgentState) -> Dict[str, Any]:
    """Evaluates student text input via Groq, assesses semantic risk, and calls models."""
    messages = state.get("messages", [])
    last_user_message = next((m.content for m in reversed(messages) if isinstance(m, HumanMessage)), "")
    
    # Grab key explicitly from environment mapping
    groq_key = os.getenv("GROQ_API_KEY")
    if not groq_key:
         raise ValueError("❌ GROQ_API_KEY is missing from system environment variables entirely.")

    # Active stable evaluator model setup
    evaluator_llm = ChatGroq(
        model="llama-3.1-8b-instant", 
        temperature=0.0,
        groq_api_key=groq_key
    ).with_structured_output(QueryEvaluation)
    
    evaluation = await evaluator_llm.ainvoke([
        {"role": "system", "content": "Analyze the incoming student inquiry for risk classification metrics. Output valid JSON matching schema definitions exactly."},
        {"role": "user", "content": last_user_message}
    ])
    
    # Fetch matching RAG context natively from local vector index
    context = get_relevant_context(query=last_user_message)
    
    system_prompt = (
        "You are an official Campus Help Desk Agent. You must answer student questions accurately based "
        "only on the provided context below. Do not guess or hallucinate. Keep answers concise.\n\n"
        f"Context:\n{context}"
    )
    
    # Active flagship generation model setup
    llm = ChatGroq(
        model="llama-3.3-70b-versatile", 
        temperature=0.1,
        groq_api_key=groq_key
    )
    formatted_messages = [{"role": "system", "content": system_prompt}] + messages
    response = await llm.ainvoke(formatted_messages)
    
    return {
        "messages": [response],
        "context": context,
        "requires_approval": evaluation.is_sensitive,
        "category": evaluation.category
    }

async def human_approval_node(state: AgentState) -> Dict[str, Any]:
    """Halts execution stack via serialized checkpoint interrupts for admin sign-off."""
    last_agent_response = state["messages"][-1].content
    
    admin_action = interrupt({
        "status": "PENDING_REVIEW",
        "category": state.get("category", "General"),
        "proposed_response": last_agent_response
    })
    
    if admin_action.get("approved", False):
        return {"approval_granted": True}
    
    return {
        "approval_granted": False,
        "messages": [AIMessage(content="Your request involves regulatory campus parameters. To guarantee accuracy, please address the administrative terminal desk directly.")]
    }