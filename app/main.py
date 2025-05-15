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

# app/main.py - Update the openai_compatible_chat function

@app.post("/v1/chat/completions")
async def openai_compatible_chat(request: Request):
    data = await request.json()
    
    messages = data.get("messages", [])
    stream = data.get("stream", False)
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
    
    # Get relevant tools for the query using our tool registry
    registry = get_tool_registry()
    relevant_tools = await get_relevant_tools(query, registry)
    
    # Transform tools to OpenAI function calling format
    tools_format = []
    for tool in relevant_tools:
        # Format arguments as OpenAI parameters schema
        properties = {}
        required = []
        
        for arg_name, arg_info in tool.get("arguments", {}).items():
            properties[arg_name] = {
                "type": arg_info.get("type", "string"),
                "description": arg_info.get("description", "")
            }
            if arg_info.get("required", False):
                required.append(arg_name)
        
        tools_format.append({
            "type": "function",
            "function": {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required
                }
            }
        })
    
    # Get LLM client
    model_name = data.get("model", os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b"))
    llm = get_llm_client(model_name)
    
    if system_message:
        llm.set_system_prompt(system_message)
    
    # Add tool information to the prompt
    tools_prompt = ""
    if tools_format:
        tools_prompt = "You have access to the following tools:\n"
        for tool in relevant_tools:
            args_info = ", ".join([f"{k}" for k in tool.get("arguments", {})])
            tools_prompt += f"- {tool['name']}({args_info}): {tool['description']}\n"
        
        tools_prompt += "\nIf the user's request can be addressed using these tools, describe how you would use them."
    
    enhanced_prompt = f"{query}\n\n{tools_prompt}"
    
    # Handle streaming
    if stream:
        async def generate():
            # Start stream with an initial chunk
            chunk_id = f"chatcmpl-{os.urandom(12).hex()}"
            async for chunk in llm.generate_response_stream(enhanced_prompt):
                chunk_data = {
                    "id": chunk_id,
                    "object": "chat.completion.chunk",
                    "created": int(time.time()),
                    "model": model_name,
                    "choices": [
                        {
                            "index": 0,
                            "delta": {
                                "content": chunk
                            },
                            "finish_reason": None
                        }
                    ]
                }
                yield f"data: {json.dumps(chunk_data)}\n\n"
            
            # End stream with a final chunk
            final_data = {
                "id": chunk_id,
                "object": "chat.completion.chunk",
                "created": int(time.time()),
                "model": model_name,
                "choices": [
                    {
                        "index": 0,
                        "delta": {},
                        "finish_reason": "stop"
                    }
                ]
            }
            yield f"data: {json.dumps(final_data)}\n\n"
            yield "data: [DONE]\n\n"
        
        return StreamingResponse(generate(), media_type="text/event-stream")
    else:
        # Generate response as normal
        response_text = await llm.generate_response(enhanced_prompt)
        
        # Return OpenAI-compatible format with tools
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
                "prompt_tokens": len(enhanced_prompt),
                "completion_tokens": len(response_text),
                "total_tokens": len(enhanced_prompt) + len(response_text)
            },
            "tools": tools_format  # This is key for OpenWebUI to show tools
        }

# Helper function to get relevant tools for a query using TF-IDF-based similarity (O(n) complexity)
async def get_relevant_tools(query: str, registry, max_tools: int = 5) -> list:
    """Get the most relevant tools for a query using TF-IDF based matching.
    This is a simplified implementation for the quick-start demo with O(n) complexity."""
    tools = registry.list_tools()
    if not tools:
        return []
    
    # Simple term frequency calculation
    query_terms = set(query.lower().split())
    
    # Score tools based on term overlap (TF-IDF simplified)
    scored_tools = []
    for tool in tools:
        tool_text = f"{tool['name']} {tool['description']}".lower()
        tool_terms = set(tool_text.split())
        
        # Calculate Jaccard similarity (intersection over union)
        intersection = len(query_terms.intersection(tool_terms))
        union = len(query_terms.union(tool_terms))
        similarity = intersection / max(1, union)  # Avoid division by zero
        
        scored_tools.append((similarity, tool))
    
    # Sort by score and take top N tools
    scored_tools.sort(reverse=True, key=lambda x: x[0])
    return [tool for _, tool in scored_tools[:max_tools]]

# Ollama-compatible endpoint for OpenWebUI
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

# @app.get("/health")
# async def health_check():
#     """Health check endpoint for monitoring."""
#     health_status = {
#         "status": "healthy",
#         "components": {}
#     }
    
#     # Check Redis connection
#     try:
#         state_manager = get_state_manager()
#         redis_ok = await state_manager.redis_client.ping()
#         health_status["components"]["redis"] = "connected" if redis_ok else "disconnected"
#     except Exception as e:
#         health_status["components"]["redis"] = f"error: {str(e)}"
    
#     # Get tool count
#     try:
#         registry = get_tool_registry()
#         tools = registry.list_tools()
#         health_status["components"]["tools"] = f"{len(tools)} tools available"
#     except Exception as e:
#         health_status["components"]["tools"] = f"error: {str(e)}"
    
#     # Overall status depends on components
#     if "error" in health_status["components"].get("redis", ""):
#         health_status["status"] = "degraded"
    
#     return health_status

# Get the existing MCP server instance
mcp_server = get_default_mcp_server()
# Mount it at /mcp path using Streamable HTTP transport
app.mount("/mcp", mcp_server.streamable_http_app())
# app.mount("/mcp", mcp_server.sse_app())
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