from contextlib import asynccontextmanager
from collections.abc import AsyncIterator
from dataclasses import dataclass
from typing import Dict, Any, Optional, List

from mcp.server.fastmcp import FastMCP, Context

from app.core.redis import get_state_manager

@dataclass
class AppContext:
    """Application context for the MCP server."""
    redis_client: Any
    tools_metadata: Dict[str, Any]
    # Add other shared resources here

@asynccontextmanager
async def app_lifespan(server: FastMCP) -> AsyncIterator[AppContext]:
    """Manage application lifecycle for the MCP server."""
    # Initialize services on startup
    state_manager = get_state_manager()
    await state_manager.initialize()
    
    # Create application context
    try:
        yield AppContext(
            redis_client=state_manager.redis_client,
            tools_metadata={}
        )
    finally:
        # Cleanup on shutdown
        await state_manager.shutdown()

class MCPServerManager:
    """Manager for MCP server instances."""
    
    _instance = None
    
    def __new__(cls):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(MCPServerManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize only once due to singleton pattern."""
        if not hasattr(self, 'initialized') or not self.initialized:
            self.mcp_servers = {}
            self.initialized = False
    
    def create_server(self, name: str) -> FastMCP:
        """
        Create a new MCP server.
        
        Args:
            name: Server name
            
        Returns:
            The created server instance
        """
        if name in self.mcp_servers:
            return self.mcp_servers[name]
        
        # Create a new MCP server
        server = FastMCP(
            name,
            lifespan=app_lifespan,
            dependencies=[
                "fastapi", 
                "redis", 
                "openai", 
                "anthropic", 
                "pinecone-client"
            ]
        )
        
        # Store the server
        self.mcp_servers[name] = server
        
        return server
    
    def get_server(self, name: str) -> Optional[FastMCP]:
        """
        Get an existing MCP server.
        
        Args:
            name: Server name
            
        Returns:
            The server instance, or None if not found
        """
        return self.mcp_servers.get(name)
    
    def list_servers(self) -> List[str]:
        """
        List all available MCP servers.
        
        Returns:
            List of server names
        """
        return list(self.mcp_servers.keys())

# Global singleton instance
_mcp_server_manager = MCPServerManager()

def get_mcp_server_manager() -> MCPServerManager:
    """Get the global MCP server manager instance."""
    return _mcp_server_manager

def get_default_mcp_server() -> FastMCP:
    """
    Get the default MCP server.
    
    Creates the server if it doesn't exist.
    
    Returns:
        The default MCP server instance
    """
    manager = get_mcp_server_manager()
    default_server_name = "rag-mcp"
    
    server = manager.get_server(default_server_name)
    if not server:
        server = manager.create_server(default_server_name)
    
    return server