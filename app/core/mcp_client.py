from typing import Dict, Any, List, Optional, Tuple
import os
import asyncio
import logging
from dataclasses import dataclass

from mcp import ClientSession, types
from mcp.client.streamable_http import streamablehttp_client
from mcp.client.stdio import stdio_client
from mcp.client import StdioServerParameters

logger = logging.getLogger(__name__)

@dataclass
class MCPServerConfig:
    """Configuration for an MCP server connection."""
    name: str
    transport_type: str  # "stdio" or "http"
    # For stdio
    command: Optional[str] = None
    args: Optional[List[str]] = None
    env: Optional[Dict[str, str]] = None
    # For HTTP
    url: Optional[str] = None

class MCPClientManager:
    """Manages multiple MCP client connections."""
    
    _instance = None
    
    def __new__(cls):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(MCPClientManager, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the manager."""
        if not hasattr(self, 'initialized'):
            self.clients = {}
            self.sessions = {}
            self.initialized = True
    
    async def initialize_client(self, config: MCPServerConfig) -> bool:
        """
        Initialize a client connection to an MCP server.
        
        Args:
            config: MCP server configuration
            
        Returns:
            True if successful
        """
        name = config.name
        
        # Check if already initialized
        if name in self.sessions and self.sessions[name]:
            return True
        
        try:
            if config.transport_type == "stdio":
                if not config.command:
                    raise ValueError("Command is required for stdio transport")
                
                # Create stdio connection
                server_params = StdioServerParameters(
                    command=config.command,
                    args=config.args or [],
                    env=config.env or {}
                )
                
                read_stream, write_stream = await stdio_client(server_params)
                
            elif config.transport_type == "http":
                if not config.url:
                    raise ValueError("URL is required for HTTP transport")
                
                # Create HTTP connection
                read_stream, write_stream, _ = await streamablehttp_client(config.url)
                
            else:
                raise ValueError(f"Unsupported transport type: {config.transport_type}")
            
            # Create session
            session = ClientSession(read_stream, write_stream)
            await session.initialize()
            
            # Store client info
            self.sessions[name] = session
            self.clients[name] = {
                "config": config,
                "tools": await session.list_tools()
            }
            
            logger.info(f"Connected to MCP server '{name}' with {len(self.clients[name]['tools'])} tools")
            return True
            
        except Exception as e:
            logger.error(f"Failed to connect to MCP server '{name}': {e}")
            return False
    
    async def close_client(self, name: str) -> bool:
        """
        Close a client connection.
        
        Args:
            name: Client name
            
        Returns:
            True if successful
        """
        if name in self.sessions and self.sessions[name]:
            try:
                await self.sessions[name].close()
                del self.sessions[name]
                if name in self.clients:
                    del self.clients[name]
                return True
            except Exception as e:
                logger.error(f"Error closing MCP client '{name}': {e}")
        return False
    
    async def close_all(self):
        """Close all client connections."""
        for name in list(self.sessions.keys()):
            await self.close_client(name)
    
    async def call_tool(self, client_name: str, tool_name: str, arguments: Dict[str, Any]) -> Any:
        """
        Call a tool on an MCP server.
        
        Args:
            client_name: MCP client name
            tool_name: Tool name
            arguments: Tool arguments
            
        Returns:
            Tool result
        """
        if client_name not in self.sessions:
            raise ValueError(f"MCP client '{client_name}' not initialized")
        
        session = self.sessions[client_name]
        result = await session.call_tool(tool_name, arguments)
        return result
    
    async def read_resource(self, client_name: str, resource_uri: str) -> Tuple[bytes, Optional[str]]:
        """
        Read a resource from an MCP server.
        
        Args:
            client_name: MCP client name
            resource_uri: Resource URI
            
        Returns:
            Tuple of (content, mime_type)
        """
        if client_name not in self.sessions:
            raise ValueError(f"MCP client '{client_name}' not initialized")
        
        session = self.sessions[client_name]
        content, mime_type = await session.read_resource(resource_uri)
        return content, mime_type
    
    def get_tools(self, client_name: str = None) -> List[Dict[str, Any]]:
        """
        Get tools from one or all MCP clients.
        
        Args:
            client_name: Optional client name to filter
            
        Returns:
            List of tool definitions
        """
        if client_name:
            if client_name not in self.clients:
                return []
            return [
                {
                    "name": tool.name,
                    "description": tool.description or "",
                    "client": client_name,
                    "arguments": self._format_tool_arguments(tool.arguments or [])
                }
                for tool in self.clients[client_name]["tools"]
            ]
        else:
            # Get tools from all clients
            all_tools = []
            for client_name, client in self.clients.items():
                all_tools.extend([
                    {
                        "name": tool.name,
                        "description": tool.description or "",
                        "client": client_name,
                        "arguments": self._format_tool_arguments(tool.arguments or [])
                    }
                    for tool in client["tools"]
                ])
            return all_tools
    
    def _format_tool_arguments(self, arguments: List[Any]) -> Dict[str, Dict[str, Any]]:
        """Format tool arguments for easier use."""
        result = {}
        for arg in arguments:
            result[arg.name] = {
                "type": arg.type or "string",
                "description": arg.description or "",
                "required": arg.required or False
            }
        return result
    
    def find_client_for_tool(self, tool_name: str) -> Optional[str]:
        """
        Find the client that has the given tool.
        
        Args:
            tool_name: Tool name
            
        Returns:
            Client name or None if not found
        """
        for client_name, client in self.clients.items():
            if any(tool.name == tool_name for tool in client["tools"]):
                return client_name
        return None

# Global singleton instance
_mcp_client_manager = MCPClientManager()

def get_mcp_client_manager() -> MCPClientManager:
    """Get the global MCP client manager instance."""
    return _mcp_client_manager

# Helper function to create and initialize a client
async def create_mcp_client(
    name: str,
    transport_type: str,
    command: str = None,
    args: List[str] = None,
    env: Dict[str, str] = None,
    url: str = None
) -> bool:
    """
    Create and initialize an MCP client.
    
    Args:
        name: Client name
        transport_type: Transport type ("stdio" or "http")
        command: Command for stdio transport
        args: Arguments for stdio transport
        env: Environment variables for stdio transport
        url: URL for HTTP transport
        
    Returns:
        True if successful
    """
    manager = get_mcp_client_manager()
    config = MCPServerConfig(
        name=name,
        transport_type=transport_type,
        command=command,
        args=args,
        env=env,
        url=url
    )
    return await manager.initialize_client(config)