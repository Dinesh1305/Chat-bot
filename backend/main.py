import json
import asyncio
from fastapi import FastAPI
from fastapi.responses import StreamingResponse
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from langchain_core.messages import HumanMessage
from .graph_core import graph_app  # Import the compiled LangGraph agent

app = FastAPI(title="LangGraph Chatbot API")

# Setup CORS to allow the Streamlit frontend to connect (default port 8501)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:8501", "http://127.0.0.1:8501"],  # Add Streamlit Cloud URL if deploying
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


class ChatRequest(BaseModel):
    """Pydantic model for incoming chat requests."""
    user_query: str
    thread_id: str  # Unique ID for LangGraph to manage conversation memory


@app.post("/chat/stream")
async def stream_chat_response(req: ChatRequest):
    """
    Asynchronously streams the LLM response from the LangGraph agent.
    """
    # LangGraph configuration for session memory
    config = {"configurable": {"thread_id": req.thread_id}}

    async def event_generator():
        try:
            # Prepare the initial input message
            input_message = {"messages": [HumanMessage(content=req.user_query)]}

            # Use LangGraph's astream_events for token streaming
            async for event in graph_app.astream_events(
                    input_message,
                    config=config,
                    version="v1"
            ):
                # Filter for the actual streaming tokens
                if event["event"] == "on_chat_model_stream":
                    chunk = event["data"]["chunk"].content
                    if chunk:
                        # Send chunk as an SSE event (Streamlit can read this)
                        yield f"data: {json.dumps({'content': chunk})}\n\n"
                        await asyncio.sleep(0)  # Yield control

                # Optional: Handle tool_calls or function_calls events here for complex agents

            # Signal the end of the stream
            yield "data: [DONE]\n\n"

        except Exception as e:
            print(f"Streaming error: {e}")
            yield f"data: {json.dumps({'error': str(e)})}\n\n"
            yield "data: [DONE]\n\n"

    # Return the StreamingResponse with the correct media type for SSE
    return StreamingResponse(event_generator(), media_type="text/event-stream")

# Command to run the backend:
# uvicorn backend.main:app --reload --host 0.0.0.0 --port 8000