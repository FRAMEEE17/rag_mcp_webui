import os
import asyncio
from fastapi import FastAPI, Depends, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, StreamingResponse
import time
import json

from app.core.mcp_server import get_default_mcp_server
from app.core.registry import get_tool_registry
from app.llm.client import get_llm_client

# Create FastAPI app
app = FastAPI(title="RAG-MCP API", description="Retrieval-Augmented Generation with Model Control Protocol")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

@app.on_event("startup")
async def startup_event():
    # Initialize MCP server
    mcp_server = get_default_mcp_server()
    
    # Initialize tool registry with MCP server
    registry = get_tool_registry()
    registry.initialize(mcp_server)
    
    # Discover and register tools
    registry.discover_tools()
    
    print(f"Initialized MCP server with {len(registry.list_tools())} tools")

@app.get("/")
async def root():
    return {"message": "RAG-MCP API is running"}

@app.get("/tools")
async def list_tools():
    registry = get_tool_registry()
    return {"tools": registry.list_tools()}

# OpenAI-compatible endpoint for OpenWebUI integration
@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Request):
    data = await request.json()
    
    messages = data.get("messages", [])
    if not messages:
        return {"error": "No messages provided"}
    
    # Extract the last user message
    last_message = messages[-1]
    if last_message.get("role") != "user":
        return {"error": "Last message must be from user"}
    
    query = last_message.get("content", "")
    
    # Get system message if available
    system_message = None
    for msg in messages:
        if msg.get("role") == "system":
            system_message = msg.get("content")
            break
    
    # Get LLM client (NVIDIA Gemma)
    model_name = data.get("model", os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b"))
    llm = get_llm_client(model_name)
    
    if system_message:
        llm.set_system_prompt(system_message)
    
    # Generate response
    response_text = await llm.generate_response(query)
    
    # Format in OpenAI-compatible format
    return {
        "id": "chatcmpl-" + os.urandom(12).hex(),
        "object": "chat.completion",
        "created": int(time.time()),
        "model": model_name,
        "choices": [
            {
                "index": 0,
                "message": {
                    "role": "assistant",
                    "content": response_text
                },
                "finish_reason": "stop"
            }
        ],
        "usage": {
            "prompt_tokens": len(query),
            "completion_tokens": len(response_text),
            "total_tokens": len(query) + len(response_text)
        }
    }

# NEW: Ollama-compatible endpoint for OpenWebUI
@app.post("/api/chat")
async def ollama_compatible_chat(request: Request):
    data = await request.json()
    
    model = data.get("model", os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b"))
    messages = data.get("messages", [])
    
    if not messages:
        return {"error": "No messages provided"}
    
    # Get LLM client
    llm = get_llm_client(model)
    
    # Set system prompt if present
    for msg in messages:
        if msg.get("role") == "system":
            llm.set_system_prompt(msg.get("content", ""))
            break
    
    # Get last user message
    user_messages = [m for m in messages if m.get("role") == "user"]
    if not user_messages:
        return {"error": "No user message found"}
    
    last_user_message = user_messages[-1]["content"]
    
    # Generate response
    response_text = await llm.generate_response(last_user_message)
    
    # Ollama-compatible response format
    return {
        "model": model,
        "created_at": int(time.time()),
        "message": {
            "role": "assistant",
            "content": response_text
        },
        "done": True
    }

# Ollama-compatible model list endpoint for OpenWebUI
@app.get("/api/tags")
async def ollama_models():
    """Return available models in Ollama-compatible format"""
    return {
        "models": [
            {
                "name": "gemma-7b",
                "model": "google/gemma-7b",
                "modified_at": int(time.time()),
                "size": 7000000000,
                "digest": "nvidia-nemo",
                "details": {
                    "format": "gguf",
                    "family": "gemma",
                    "families": ["gemma", "google"],
                    "parameter_size": "7B",
                    "quantization_level": "none"
                }
            }
        ]
    }

# Mount MCP application
from mcp.server.fastmcp import get_app as get_mcp_app
app.mount("/mcp", get_mcp_app(get_default_mcp_server()))

# Create a static directory for web interface
os.makedirs("static", exist_ok=True)

# Mount static files directory
app.mount("/static", StaticFiles(directory="static"), name="static")

# Add route to serve the demo interface
@app.get("/demo")
async def demo():
    return FileResponse("static/index.html")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run("app.main:app", host="0.0.0.0", port=8000, reload=True)