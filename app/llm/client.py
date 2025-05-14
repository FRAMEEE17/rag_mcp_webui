import os
import json
import httpx
from typing import Dict, Any, Optional, List

class LLMClient:
    """Generic LLM client interface."""
    
    def __init__(self, model_name: str):
        """Initialize LLM client with model name."""
        self.model_name = model_name
        self.system_prompt = "You are a helpful AI assistant."
    
    def set_system_prompt(self, prompt: str):
        """Set the system prompt for the LLM."""
        self.system_prompt = prompt
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from the LLM."""
        raise NotImplementedError("Subclasses must implement generate_response")

class NvidiaGemmaClient(LLMClient):
    """Client for NVIDIA API (Gemma model)."""
    
    def __init__(self, model_name: str = "google/gemma-7b"):
        super().__init__(model_name)
        self.api_key = os.getenv("NVIDIA_API_KEY")
        if not self.api_key:
            raise ValueError("NVIDIA_API_KEY environment variable is required")
        self.api_url = "https://integrate.api.nvidia.com/v1"
    
    async def generate_response(self, prompt: str) -> str:
        """Generate a response from NVIDIA-hosted Gemma."""
        from openai import AsyncOpenAI
        
        client = AsyncOpenAI(
            base_url=self.api_url,
            api_key=self.api_key
        )
        
        try:
            response = await client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": self.system_prompt},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.5,
                top_p=1,
                max_tokens=1024
            )
            
            return response.choices[0].message.content
        except Exception as e:
            print(f"Error calling NVIDIA API: {str(e)}")
            raise

# Factory function to get LLM client
def get_llm_client(model_name: str = None) -> LLMClient:
    """Get an LLM client based on model name or environment variables."""
    if not model_name:
        model_name = os.getenv("DEFAULT_LLM_MODEL", "google/gemma-7b")
    
    # Determine provider from model name
    if model_name.startswith("google/gemma"):
        return NvidiaGemmaClient(model_name)
    else:
        # Fallback to other provider if needed
        return NvidiaGemmaClient(model_name)  