from typing import Dict, Any, Callable, List, Optional, Type
import inspect
import importlib
import os
import glob
from functools import wraps
from dataclasses import dataclass

from mcp.server.fastmcp import FastMCP

@dataclass
class ToolMetadata:
    """Metadata for a registered tool."""
    name: str
    function: Callable
    description: str
    category: str
    arguments: Dict[str, Dict[str, Any]]

class ToolRegistry:
    """Registry for MCP tools with automatic discovery."""
    _instance = None
    
    def __new__(cls):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(ToolRegistry, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize only once due to singleton pattern."""
        if not hasattr(self, 'initialized'):
            self.tools: Dict[str, ToolMetadata] = {}
            self.mcp_server: Optional[FastMCP] = None
            self.initialized = True
    
    def initialize(self, mcp_server: FastMCP):
        self.mcp_server = mcp_server
    
    def register_tool(
        self, 
        func: Callable = None, 
        *, 
        name: str = None, 
        description: str = None,
        category: str = "general"
    ) -> Callable:
        """
        Register a function as a tool.
        
        Can be used as a decorator:
        @registry.register_tool(name="my_tool", description="My tool")
        def my_tool(): ...
        
        Or directly:
        registry.register_tool(my_tool, name="my_tool", description="My tool")
        
        Args:
            func: The function to register
            name: Optional custom name
            description: Optional description
            category: Tool category for organization
            
        Returns:
            The original function (for decorator usage)
        """
        def decorator(func: Callable) -> Callable:
            # Get function metadata
            tool_name = name or func.__name__
            tool_description = description or func.__doc__ or ""
            
            # Extract argument info from function signature
            signature = inspect.signature(func)
            arguments = {}
            
            for param_name, param in signature.parameters.items():
                # Skip self, cls, and context parameters
                if param_name in ('self', 'cls', 'ctx', 'context'):
                    continue
                
                param_type = param.annotation if param.annotation != inspect.Parameter.empty else Any
                param_type_name = getattr(param_type, '__name__', str(param_type))
                
                # Handle default values
                has_default = param.default != inspect.Parameter.empty
                default_value = param.default if has_default else None
                
                arguments[param_name] = {
                    'type': param_type_name,
                    'description': '',  # Could extract from docstring
                    'required': not has_default,
                    'default': default_value if has_default else None
                }
            
            # Store tool metadata
            self.tools[tool_name] = ToolMetadata(
                name=tool_name,
                function=func,
                description=tool_description.strip(),
                category=category,
                arguments=arguments
            )
            
            # Register with MCP server 
            if self.mcp_server:
                self.mcp_server.add_tool(
                    func, 
                    name=tool_name, 
                    description=tool_description
                )
            
            # Preserve function signature and docstring
            return func
        
        # Handle both @register_tool and @register_tool() usage
        if func is None:
            return decorator
        return decorator(func)
    
    def discover_tools(self, tools_dir: str = "app/tools"):
        """
        Automatically discover and register tools.
        
        Args:
            tools_dir: Directory containing tool modules
        """
        if not self.mcp_server:
            raise RuntimeError("Tool registry not initialized with MCP server")
        
        # Get Python files in tools directory
        tool_files = glob.glob(f"{tools_dir}/*.py")
        
        for file_path in tool_files:
            filename = os.path.basename(file_path)
            
            # Skip base and special files
            if filename == "base.py" or filename.startswith("__"):
                continue
            
            # Convert file path to module path
            module_path = file_path.replace("/", ".").replace("\\", ".")
            if module_path.endswith(".py"):
                module_path = module_path[:-3]
            
            try:
                # Import the module
                module = importlib.import_module(module_path)
                
                # Look for tool functions
                for name, obj in inspect.getmembers(module):
                    if inspect.isfunction(obj) and hasattr(obj, "_is_tool"):
                        # Register the tool
                        self.register_tool(
                            obj,
                            name=getattr(obj, "_tool_name", name),
                            description=getattr(obj, "_tool_description", obj.__doc__),
                            category=getattr(obj, "_tool_category", "general")
                        )
            except ImportError as e:
                print(f"Failed to import module {module_path}: {e}")
        
        print(f"Discovered {len(self.tools)} tools")
    
    def get_tool(self, name: str) -> Optional[ToolMetadata]:
        """
        Get a registered tool by name.
        
        Args:
            name: Tool name
            
        Returns:
            Tool metadata or None if not found
        """
        return self.tools.get(name)
    
    def list_tools(self, category: str = None) -> List[Dict[str, Any]]:
        """
        List all registered tools.
        
        Args:
            category: Optional category filter
            
        Returns:
            List of tool information
        """
        tools = list(self.tools.values())
        
        if category:
            tools = [tool for tool in tools if tool.category == category]
        
        return [
            {
                "name": tool.name,
                "description": tool.description,
                "category": tool.category,
                "arguments": tool.arguments
            }
            for tool in tools
        ]
    
    def list_categories(self) -> List[str]:
        """
        Get all unique tool categories.
        
        Returns:
            List of category names
        """
        return list(set(tool.category for tool in self.tools.values()))

# Global singleton instance
_tool_registry = ToolRegistry()

def get_tool_registry() -> ToolRegistry:
    """Get the global tool registry instance."""
    return _tool_registry

# Tool decorator for use in tool modules
def tool(name: str = None, description: str = None, category: str = "general"):
    """
    Decorator to mark a function as a tool.
    
    Args:
        name: Tool name (default: function name)
        description: Tool description (default: function docstring)
        category: Tool category (default: "general")
        
    Returns:
        Decorated function
    """
    def decorator(func: Callable) -> Callable:
        @wraps(func)
        def wrapper(*args, **kwargs):
            return func(*args, **kwargs)
        
        # Mark as a tool for discovery
        wrapper._is_tool = True
        wrapper._tool_name = name
        wrapper._tool_description = description
        wrapper._tool_category = category
        
        # Preserve signature and docstring
        wrapper.__signature__ = inspect.signature(func)
        if func.__doc__:
            wrapper.__doc__ = func.__doc__
        
        return wrapper
    
    return decorator