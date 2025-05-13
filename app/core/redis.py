import redis.asyncio as redis
import json
from typing import Any, Dict, Optional
import os
import pickle

class StateManager:
    """Redis-based state management for the application."""
    
    _instance = None
    
    def __new__(cls):
        """Implement the singleton pattern."""
        if cls._instance is None:
            cls._instance = super(StateManager, cls).__new__(cls)
            cls._instance.initialized = False
        return cls._instance
    
    def __init__(self):
        """Initialize only once due to singleton pattern."""
        if not hasattr(self, 'initialized') or not self.initialized:
            self.redis_url = os.getenv("REDIS_URL", "redis://localhost:6379/0")
            self.redis_client = None
            self.initialized = False
    
    async def initialize(self):
        """Initialize Redis connection."""
        if not self.initialized:
            self.redis_client = redis.from_url(self.redis_url)
            # Test connection
            ping = await self.redis_client.ping()
            if not ping:
                raise ConnectionError(f"Could not connect to Redis at {self.redis_url}")
            self.initialized = True
    
    async def shutdown(self):
        """Close Redis connection."""
        if self.redis_client:
            await self.redis_client.close()
            self.initialized = False
    
    async def set(self, key: str, value: Any, ttl: Optional[int] = None) -> bool:
        """
        Set a value in Redis.
        
        Args:
            key: Redis key
            value: Value to store (will be serialized)
            ttl: Time-to-live in seconds (optional)
            
        Returns:
            True if successful
        """
        if not self.initialized:
            await self.initialize()
        
        # Handle different types of values
        if isinstance(value, (dict, list, tuple, set)):
            # JSON serializable data
            serialized = json.dumps(value)
            value_type = "json"
        else:
            # Use pickle for non-JSON serializable data
            serialized = pickle.dumps(value)
            value_type = "pickle"
        
        # Store with type information
        pipeline = self.redis_client.pipeline()
        pipeline.set(f"{key}:type", value_type)
        pipeline.set(key, serialized)
        
        if ttl:
            pipeline.expire(key, ttl)
            pipeline.expire(f"{key}:type", ttl)
        
        await pipeline.execute()
        return True
    
    async def get(self, key: str, default: Any = None) -> Any:
        """
        Get a value from Redis.
        
        Args:
            key: Redis key
            default: Default value if key doesn't exist
            
        Returns:
            The stored value, or default if not found
        """
        if not self.initialized:
            await self.initialize()
        
        # Get value and its type
        pipeline = self.redis_client.pipeline()
        pipeline.get(key)
        pipeline.get(f"{key}:type")
        
        value, value_type = await pipeline.execute()
        
        if value is None:
            return default
        
        # Deserialize based on type
        if value_type == b"json":
            return json.loads(value)
        elif value_type == b"pickle":
            return pickle.loads(value)
        else:
            # Fallback: try JSON first, then pickle
            try:
                return json.loads(value)
            except:
                try:
                    return pickle.loads(value)
                except:
                    return value
    
    async def delete(self, key: str) -> bool:
        """
        Delete a value from Redis.
        
        Args:
            key: Redis key
            
        Returns:
            True if successful
        """
        if not self.initialized:
            await self.initialize()
        
        pipeline = self.redis_client.pipeline()
        pipeline.delete(key)
        pipeline.delete(f"{key}:type")
        
        results = await pipeline.execute()
        return results[0] > 0

# Global singleton instance
_state_manager = StateManager()

def get_state_manager() -> StateManager:
    """Get the global state manager instance."""
    return _state_manager