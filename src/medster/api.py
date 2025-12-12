"""
FastAPI backend for Medster-local-LLM web interface.
Provides WebSocket streaming and REST endpoints for the medical analysis agent.
"""
from fastapi import FastAPI, WebSocket, WebSocketDisconnect, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, List, Dict, Any
import asyncio
import json
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

from medster.agent import Agent
from medster import config

app = FastAPI(title="Medster Local LLM API", version="1.0.0")

# CORS middleware for local development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://127.0.0.1:3000"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state for selected model
current_model = "gpt-oss:20b"


# Pydantic models
class ModelSelection(BaseModel):
    model_name: str


class ChatMessage(BaseModel):
    message: str
    model: Optional[str] = None


class ModelInfo(BaseModel):
    name: str
    description: str
    multimodal: bool


# Available models
AVAILABLE_MODELS = [
    ModelInfo(
        name="gpt-oss:20b",
        description="Text-only model - Faster inference for clinical reasoning, labs, notes, reports",
        multimodal=False
    ),
    ModelInfo(
        name="qwen3-vl:8b",
        description="Multimodal vision support - Can analyze DICOM images, ECG tracings, X-rays",
        multimodal=True
    ),
    ModelInfo(
        name="ministral-3:8b",
        description="Multimodal vision support - Alternative vision model with strong reasoning capabilities",
        multimodal=True
    )
]


@app.get("/")
async def root():
    """Health check endpoint."""
    return {
        "status": "online",
        "service": "Medster Local LLM API",
        "current_model": current_model
    }


@app.get("/api/models", response_model=List[ModelInfo])
async def get_models():
    """Get list of available Ollama models."""
    return AVAILABLE_MODELS


@app.post("/api/select-model")
async def select_model(selection: ModelSelection):
    """Set the active Ollama model."""
    global current_model
    
    # Validate model exists
    valid_models = [m.name for m in AVAILABLE_MODELS]
    if selection.model_name not in valid_models:
        raise HTTPException(
            status_code=400,
            detail=f"Invalid model. Choose from: {', '.join(valid_models)}"
        )
    
    current_model = selection.model_name
    config.set_selected_model(current_model)
    
    return {
        "status": "success",
        "model": current_model,
        "message": f"Model set to {current_model}"
    }


@app.get("/api/current-model")
async def get_current_model():
    """Get the currently selected model."""
    return {
        "model": current_model,
        "info": next((m for m in AVAILABLE_MODELS if m.name == current_model), None)
    }


class StreamingCallback:
    """Callback handler for streaming agent events to WebSocket."""

    def __init__(self, websocket: WebSocket):
        self.websocket = websocket
        self.loop = asyncio.get_event_loop()
        self.connected = True

    def disconnect(self):
        """Mark the WebSocket as disconnected to stop sending events."""
        self.connected = False

    async def send_event(self, event_type: str, data: Any):
        """Send an event to the WebSocket client."""
        if not self.connected:
            return  # Silently skip if disconnected

        try:
            await self.websocket.send_json({
                "type": event_type,
                "data": data
            })
        except Exception as e:
            # Connection closed - stop sending events
            self.connected = False
    
    def on_task_start(self, task_description: str):
        """Called when a new task starts."""
        asyncio.run_coroutine_threadsafe(
            self.send_event("task_start", {"task": task_description}),
            self.loop
        )
    
    def on_tool_execution(self, tool_name: str, args: Dict, result: Any):
        """Called when a tool is executed."""
        asyncio.run_coroutine_threadsafe(
            self.send_event("tool_execution", {
                "tool": tool_name,
                "args": args,
                "result": str(result)[:500]  # Truncate large results
            }),
            self.loop
        )
    
    def on_task_complete(self, task_description: str):
        """Called when a task completes."""
        asyncio.run_coroutine_threadsafe(
            self.send_event("task_complete", {"task": task_description}),
            self.loop
        )
    
    def on_log(self, message: str):
        """Called for general log messages."""
        asyncio.run_coroutine_threadsafe(
            self.send_event("log", {"message": message}),
            self.loop
        )
    
    def on_answer(self, answer: str):
        """Called when final answer is generated."""
        asyncio.run_coroutine_threadsafe(
            self.send_event("answer", {"answer": answer}),
            self.loop
        )


@app.websocket("/ws/chat")
async def websocket_chat(websocket: WebSocket):
    """
    WebSocket endpoint for streaming chat with the Medster agent.
    
    Client sends: {"message": "query text", "model": "optional-model-name"}
    Server streams: {"type": "event_type", "data": {...}}
    """
    await websocket.accept()
    
    try:
        while True:
            # Receive message from client
            data = await websocket.receive_json()
            message = data.get("message", "")
            model = data.get("model", current_model)
            
            if not message:
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": "Empty message received"}
                })
                continue
            
            # Send acknowledgment
            await websocket.send_json({
                "type": "start",
                "data": {
                    "message": message,
                    "model": model
                }
            })
            
            # Create callback for streaming
            callback = StreamingCallback(websocket)
            
            # Run agent in thread pool to avoid blocking
            try:
                # Create agent instance
                agent = Agent(model_name=model)
                
                # Monkey-patch the logger to stream events
                original_log = agent.logger._log
                original_log_task_start = agent.logger.log_task_start
                original_log_task_done = agent.logger.log_task_done
                original_log_tool_run = agent.logger.log_tool_run
                
                def streaming_log(msg: str):
                    original_log(msg)
                    callback.on_log(msg)
                
                def streaming_task_start(task: str):
                    original_log_task_start(task)
                    callback.on_task_start(task)
                
                def streaming_task_done(task: str):
                    original_log_task_done(task)
                    callback.on_task_complete(task)
                
                def streaming_tool_run(args, result):
                    original_log_tool_run(args, result)
                    callback.on_tool_execution("tool", args, result)
                
                agent.logger._log = streaming_log
                agent.logger.log_task_start = streaming_task_start
                agent.logger.log_task_done = streaming_task_done
                agent.logger.log_tool_run = streaming_tool_run
                
                # Run agent query in executor
                loop = asyncio.get_event_loop()
                answer = await loop.run_in_executor(None, agent.run, message)
                
                # Send final answer
                callback.on_answer(answer)
                
                # Send completion event
                await websocket.send_json({
                    "type": "complete",
                    "data": {"answer": answer}
                })
                
            except Exception as e:
                callback.disconnect()  # Stop sending events on error
                await websocket.send_json({
                    "type": "error",
                    "data": {"message": f"Agent error: {str(e)}"}
                })
    
    except WebSocketDisconnect:
        print("WebSocket client disconnected")
        # Mark callback as disconnected to prevent further event sending
        if 'callback' in locals():
            callback.disconnect()
    except Exception as e:
        print(f"WebSocket error: {e}")
        # Mark callback as disconnected to prevent further event sending
        if 'callback' in locals():
            callback.disconnect()
        try:
            await websocket.send_json({
                "type": "error",
                "data": {"message": str(e)}
            })
        except:
            pass


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
