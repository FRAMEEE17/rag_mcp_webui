import os
import json
import httpx
import asyncio
from typing import Dict, Any, Optional, List, AsyncGenerator, Callable, Union
import time

class LLMClient:
    """Generic LLM client interface with streaming support."""
    
    def __init__(self, model_name: str):
        """Initialize LLM client with model name."""
        self.model_name = model_name
        self.system_prompt = "You are a helpful AI assistant."
        self._client = None
        self._semaphore = asyncio.Semaphore(5)  # Limit concurrent requests
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the LLM."""
        self.system_prompt = prompt
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError("Subclasses must implement generate_response")
    
    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from the LLM.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Yields:
            Text chunks as they become available
        """
        # Default implementation collects the entire response and yields it as one chunk
        full_response = await self.generate_response(prompt)
        yield full_response

class NvidiaGemmaClient(LLMClient):
    """Client for NVIDIA API (Gemma model) with streaming support."""
    
    def __init__(self, model_name: str = "google/gemma-7b"):
        super().__init__(model_name)
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")
        self.api_url = "https://integrate.api.nvidia.com/v1"
        self._client = None
        self.timeout = httpx.Timeout(30.0, connect=10.0)
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def _ensure_client(self):
        """Ensure the OpenAI client is initialized (lazy loading for performance)."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                
                # Initialize with proper settings
                self._client = AsyncOpenAI(
                    base_url=self.api_url,
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from NVIDIA-hosted Gemma."""
        # Ensure client is initialized
        await self._ensure_client()
        
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        top_p=0.95,
                        max_tokens=1024
                    )
                    
                    return response.choices[0].message.content
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        print(f"Error calling NVIDIA API after {retry_count} retries: {str(e)}")
                        raise
                    
                    # Exponential backoff
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))
                    print(f"API call failed, retrying in {wait_time:.1f}s: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from NVIDIA-hosted Gemma.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Yields:
            Text chunks as they become available
        """
        # Ensure client is initialized
        await self._ensure_client()
        
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            try:
                # Create streaming request
                stream = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    top_p=0.95,
                    max_tokens=1024,
                    stream=True
                )
                
                # Process streaming response
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                print(f"Error in streaming response from NVIDIA API: {str(e)}")
                # Fallback to non-streaming version
                print("Falling back to non-streaming response")
                full_response = await self.generate_response(prompt)
                yield full_response

class AnthropicClaudeClient(LLMClient):
    """Client for Anthropic Claude API with streaming support."""
    
    def __init__(self, model_name: str = "claude-3-opus-20240229"):
        super().__init__(model_name)
        self.api_key = os.getenv("ANTHROPIC_API_KEY")
        if not self.api_key:
            raise ValueError("ANTHROPIC_API_KEY environment variable is required")
        self.api_url = "https://api.anthropic.com/v1/messages"
        self.timeout = httpx.Timeout(120.0, connect=10.0)
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from Anthropic Claude."""
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    # Prepare request
                    headers = {
                        "x-api-key": self.api_key,
                        "anthropic-version": "2023-06-01",
                        "content-type": "application/json"
                    }
                    
                    data = {
                        "model": self.model_name,
                        "messages": [
                            {"role": "user", "content": prompt}
                        ],
                        "system": self.system_prompt,
                        "max_tokens": 4096
                    }
                    
                    # Send request
                    async with httpx.AsyncClient(timeout=self.timeout) as client:
                        response = await client.post(
                            self.api_url,
                            headers=headers,
                            json=data
                        )
                        
                        if response.status_code == 200:
                            result = response.json()
                            return result["content"][0]["text"]
                        else:
                            if retry_count + 1 >= self.max_retries:
                                print(f"API error: {response.status_code} - {response.text}")
                                raise ValueError(f"API error: {response.status_code}")
                    
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        print(f"Error calling Anthropic API after {retry_count} retries: {str(e)}")
                        raise
                    
                    # Exponential backoff
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))
                    print(f"API call failed, retrying in {wait_time:.1f}s: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from Anthropic Claude.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Yields:
            Text chunks as they become available
        """
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            try:
                # Prepare request
                headers = {
                    "x-api-key": self.api_key,
                    "anthropic-version": "2023-06-01",
                    "content-type": "application/json"
                }
                
                data = {
                    "model": self.model_name,
                    "messages": [
                        {"role": "user", "content": prompt}
                    ],
                    "system": self.system_prompt,
                    "max_tokens": 4096,
                    "stream": True
                }
                
                # Send streaming request
                async with httpx.AsyncClient(timeout=self.timeout) as client:
                    async with client.stream(
                        "POST",
                        self.api_url,
                        headers=headers,
                        json=data
                    ) as response:
                        if response.status_code != 200:
                            print(f"API error: {response.status_code}")
                            # Fallback to non-streaming
                            full_response = await self.generate_response(prompt)
                            yield full_response
                            return
                        
                        # Process the streaming response
                        buffer = ""
                        async for chunk in response.aiter_text():
                            if chunk.startswith("data: "):
                                payload = chunk[6:].strip()
                                if payload == "[DONE]":
                                    break
                                
                                try:
                                    data = json.loads(payload)
                                    if "content" in data and data["content"] and data["content"][0]["type"] == "text":
                                        text = data["content"][0]["text"]
                                        yield text
                                except json.JSONDecodeError:
                                    print(f"Failed to parse JSON: {payload}")
                                    # Collect unparseable chunks in buffer
                                    buffer += payload
                        
                        # If we have leftover buffer content, yield it
                        if buffer:
                            yield buffer
            
            except Exception as e:
                print(f"Error in streaming response from Anthropic API: {str(e)}")
                # Fallback to non-streaming version
                print("Falling back to non-streaming response")
                full_response = await self.generate_response(prompt)
                yield full_response

class OpenAIClient(LLMClient):
    """Client for OpenAI API with streaming support."""
    
    def __init__(self, model_name: str = "gpt-4o"):
        super().__init__(model_name)
        self.api_key = os.getenv("OPENAI_API_KEY")
        if not self.api_key:
            raise ValueError("OPENAI_API_KEY environment variable is required")
        self.api_url = "https://api.openai.com/v1"  # Default OpenAI API URL
        self._client = None
        self.timeout = httpx.Timeout(60.0, connect=10.0)
        self.max_retries = 3
        self.retry_delay = 1.0
    
    async def _ensure_client(self):
        """Ensure the OpenAI client is initialized."""
        if self._client is None:
            try:
                from openai import AsyncOpenAI
                
                # Initialize with proper settings
                self._client = AsyncOpenAI(
                    api_key=self.api_key,
                    timeout=self.timeout
                )
            except ImportError:
                raise ImportError("OpenAI package not installed. Install with 'pip install openai'")
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from OpenAI models."""
        # Ensure client is initialized
        await self._ensure_client()
        
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            retry_count = 0
            while retry_count < self.max_retries:
                try:
                    response = await self._client.chat.completions.create(
                        model=self.model_name,
                        messages=[
                            {"role": "system", "content": self.system_prompt},
                            {"role": "user", "content": prompt}
                        ],
                        temperature=0.7,
                        max_tokens=4096
                    )
                    
                    return response.choices[0].message.content
                except Exception as e:
                    retry_count += 1
                    if retry_count >= self.max_retries:
                        print(f"Error calling OpenAI API after {retry_count} retries: {str(e)}")
                        raise
                    
                    # Exponential backoff
                    wait_time = self.retry_delay * (2 ** (retry_count - 1))
                    print(f"API call failed, retrying in {wait_time:.1f}s: {str(e)}")
                    await asyncio.sleep(wait_time)
    
    async def generate_response_stream(self, prompt: str) -> AsyncGenerator[str, None]:
        """
        Generate a streaming response from OpenAI models.
        
        Args:
            prompt: The prompt to send to the LLM
            
        Yields:
            Text chunks as they become available
        """
        # Ensure client is initialized
        await self._ensure_client()
        
        # Use semaphore to limit concurrent requests
        async with self._semaphore:
            try:
                # Create streaming request
                stream = await self._client.chat.completions.create(
                    model=self.model_name,
                    messages=[
                        {"role": "system", "content": self.system_prompt},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.7,
                    max_tokens=4096,
                    stream=True
                )
                
                # Process streaming response
                async for chunk in stream:
                    if chunk.choices and chunk.choices[0].delta.content:
                        yield chunk.choices[0].delta.content
            except Exception as e:
                print(f"Error in streaming response from OpenAI API: {str(e)}")
                # Fallback to non-streaming version
                print("Falling back to non-streaming response")
                full_response = await self.generate_response(prompt)
                yield full_response

class LLMClientFactory:
    """Factory for creating LLM clients based on model name."""
    
    _instance = None
    
    def __new__(cls):
        """Implement singleton pattern."""
        if cls._instance is None:
            cls._instance = super(LLMClientFactory, cls).__new__(cls)
        return cls._instance
    
    def __init__(self):
        """Initialize the factory."""
        self.client_cache = {}
    
    def get_client(self, model_name: Optional[str] = None) -> LLMClient:
        if not model_name:
            model_name = os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b")
        
        # Check cache first (efficient O(1) lookup)
        if model_name in self.client_cache:
            return self.client_cache[model_name]
        
        # Create new client based on model name
        client = None
        
        if model_name.startswith("google/gemma") or model_name.startswith("gemma"):
            client = NvidiaGemmaClient(model_name)
        elif model_name.startswith("claude"):
            client = AnthropicClaudeClient(model_name)
        elif model_name.startswith("gpt-") or model_name == "gpt-4o":
            client = OpenAIClient(model_name)
        else:
            # Default to Gemma
            print(f"Unknown model '{model_name}', defaulting to Gemma")
            client = NvidiaGemmaClient(os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b"))
        
        # Cache and return (efficient O(1) lookup for future requests)
        self.client_cache[model_name] = client
        return client

# Global factory instance
_llm_client_factory = LLMClientFactory()

def get_llm_client(model_name: Optional[str] = None) -> LLMClient:

    return _llm_client_factory.get_client(model_name)