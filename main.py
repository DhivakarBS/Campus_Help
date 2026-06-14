import os
import uuid
import traceback
from fastapi import FastAPI, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from langchain_core.messages import HumanMessage
from src.graph.build import agent_gateway
from dotenv import load_dotenv

# Load variables from .env file during local execution
load_dotenv()

app = FastAPI(title="Campus-Help Agent Gateway", version="1.0.0")

# --- PRODUCTION CLOUD CONFIG: CORS MIDDLEWARE ---
# This ensures public network clients can hit your Render API routes cleanly
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# --- DATA CONTRACTS ---
class ChatPayload(BaseModel):
    thread_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    message: str

class ApprovalPayload(BaseModel):
    thread_id: str
    approved: bool

# --- UI WORKSPACE ROUTER ---
@app.get("/", response_class=HTMLResponse, tags=["Frontend Client"])
async def serve_chat_workspace():
    """Serves the real-time ChatGPT-style user interface application."""
    html_path = os.path.join(os.path.dirname(__file__), "templates", "index.html")
    if not os.path.exists(html_path):
        raise HTTPException(status_code=404, detail="Template UI source map not found.")
    with open(html_path, "r", encoding="utf-8") as file:
        return HTMLResponse(content=file.read(), status_code=200)

# --- ENGINE ENDPOINTS ---
@app.post("/api/chat", tags=["Student Hub"])
async def chat_with_agent(payload: ChatPayload):
    """Processes incoming inquiries, pulls matching RAG documents, and manages execution interrupts."""
    config = {"configurable": {"thread_id": payload.thread_id}}
    input_data = {"messages": [HumanMessage(content=payload.message)]}
    
    try:
        async for event in agent_gateway.astream(input_data, config, stream_mode="updates"):
            state = await agent_gateway.aget_state(config)
            if state.next and state.tasks and state.tasks[0].interrupts:
                interrupt_details = state.tasks[0].interrupts[0].value
                return {
                    "thread_id": payload.thread_id,
                    "status": "AWAITING_ADMIN_APPROVAL",
                    "payload": interrupt_details
                }
                
        final_state = await agent_gateway.aget_state(config)
        return {
            "thread_id": payload.thread_id,
            "status": "COMPLETED",
            "response": final_state.values["messages"][-1].content
        }
    except Exception as e:
        # Catch and explicitly dump runtime bugs to the Render / Uvicorn system logs
        print("\n❌ [BACKEND CRASH TRACEBACK] ❌")
        traceback.print_exc()
        print("──────────────────────────────────────\n")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/api/admin/approve", tags=["Admin Terminal Desk"])
async def submit_admin_verdict(payload: ApprovalPayload):
    """Resumes a paused execution engine workflow by feeding back external admin review approvals."""
    config = {"configurable": {"thread_id": payload.thread_id}}
    state = await agent_gateway.aget_state(config)
    
    if not state.next:
        raise HTTPException(status_code=400, detail="Target thread context has no active pending interrupts.")
        
    try:
        feedback_data = {"approved": payload.approved}
        async for event in agent_gateway.astream(None, config, stream_mode="updates", feedback=feedback_data):
            pass
            
        final_state = await agent_gateway.aget_state(config)
        return {
            "thread_id": payload.thread_id,
            "status": "SUCCESSFULLY_RESOLVED",
            "final_response": final_state.values["messages"][-1].content
        }
    except Exception as e:
        print("\n❌ [ADMIN OVERRIDE CRASH TRACEBACK] ❌")
        traceback.print_exc()
        print("──────────────────────────────────────\n")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    import uvicorn
    # Module target definition for local quick execution testing
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=True)