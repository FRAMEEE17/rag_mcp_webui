from app.core.registry import tool
from typing import List, Dict, Any, Optional
import json
import asyncio
import os
from app.llm.client import get_llm_client
from app.core.redis import get_state_manager

class SequentialThinkingSession:
    """Manages a sequential thinking session with LLM-generated thoughts."""
    
    def __init__(self, session_id: str, problem: str):
        """
        Initialize a sequential thinking session.
        
        Args:
            session_id: Unique session identifier
            problem: Problem statement to analyze
        """
        self.session_id = session_id
        self.problem = problem
        self.thoughts = []
        self.branches = {}
        self.current_branch_id = None
        
        # Use a separate LLM instance for thinking
        self.thinking_llm = get_llm_client(os.getenv("SEQUENTIAL_THINKING_MODEL", "google/gemma-7b"))
        
        # Set system prompt for the thinking process
        self.thinking_llm.set_system_prompt(
            """You are an advanced sequential thinking engine that excels at breaking down complex problems 
            into clear, logical steps. You think step by step, revising your thoughts when necessary 
            and branching into alternative approaches when helpful. You maintain context between steps 
            and filter out irrelevant information. Each thought should build upon previous ones, creating 
            a coherent chain of reasoning that leads to a well-reasoned conclusion."""
        )
    
    async def add_thought(self, thought_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Add a thought to the session.
        
        Args:
            thought_data: Thought data including content and metadata
            
        Returns:
            Updated session state
        """
        # Validate thought data
        self._validate_thought_data(thought_data)
        
        # Add thought to history
        self.thoughts.append(thought_data)
        
        # Handle branches
        if thought_data.get("branch_from_thought") and thought_data.get("branch_id"):
            branch_id = thought_data["branch_id"]
            if branch_id not in self.branches:
                self.branches[branch_id] = []
            self.branches[branch_id].append(thought_data)
            self.current_branch_id = branch_id
        
        # Save session state
        await self._save_session_state()
        
        return {
            "thought_number": thought_data["thought_number"],
            "total_thoughts": thought_data["total_thoughts"],
            "next_thought_needed": thought_data["next_thought_needed"],
            "branches": list(self.branches.keys()),
            "thought_history_length": len(self.thoughts)
        }
    
    async def generate_next_thought(self) -> Dict[str, Any]:
        """
        Generate the next thought in the sequence using LLM.
        
        Returns:
            Generated thought data
        """
        # Prepare context from previous thoughts
        context = self._prepare_context_for_llm()
        
        # Generate next thought
        prompt = f"""
        Problem: {self.problem}
        
        Previous thoughts:
        {context}
        
        Generate the next logical thought in this sequence. Analyze the problem step by step, 
        building on previous thoughts. If appropriate, revise earlier thoughts or branch into 
        a new approach.
        
        Format your response as a JSON object with these fields:
        - thought: Your detailed thinking step
        - thought_number: The sequential number of this thought
        - total_thoughts: Your current estimate of total thoughts needed
        - next_thought_needed: Whether another thought is needed after this one
        - is_revision: (optional) Whether this revises a previous thought
        - revises_thought: (optional) Which thought number is being revised
        - branch_from_thought: (optional) If branching, which thought number is the branching point
        - branch_id: (optional) Identifier for the branch
        - needs_more_thoughts: (optional) If more thoughts are needed beyond original estimate
        
        Respond ONLY with the JSON object, no other text.
        """
        
        # Generate thought using LLM
        response = await self.thinking_llm.generate_response(prompt)
        
        # Extract JSON from response
        try:
            thought_data = self._extract_json_from_response(response)
            return thought_data
        except Exception as e:
            # If JSON extraction fails, create a formatted thought
            return {
                "thought": f"I need to reconsider my approach. {str(e)}",
                "thought_number": len(self.thoughts) + 1,
                "total_thoughts": max(len(self.thoughts) + 2, self.thoughts[-1]["total_thoughts"] if self.thoughts else 3),
                "next_thought_needed": True
            }
    
    async def get_final_answer(self) -> str:
        """
        Generate a final answer based on the thought sequence.
        
        Returns:
            Final answer to the problem
        """
        # Prepare context from all thoughts
        context = self._prepare_context_for_llm()
        
        # Generate final answer
        prompt = f"""
        Problem: {self.problem}
        
        Thinking process:
        {context}
        
        Based on this step-by-step thinking process, provide a clear, concise final answer to the original problem.
        Your answer should synthesize the insights from the thought sequence and present a well-reasoned conclusion.
        """
        
        # Use the thinking LLM to generate the answer
        response = await self.thinking_llm.generate_response(prompt)
        
        return response
    
    def _validate_thought_data(self, thought_data: Dict[str, Any]) -> None:
        """Validate required fields in thought data."""
        required_fields = ["thought", "thought_number", "total_thoughts", "next_thought_needed"]
        for field in required_fields:
            if field not in thought_data:
                raise ValueError(f"Missing required field: {field}")
        
        if not isinstance(thought_data["thought"], str):
            raise ValueError("'thought' must be a string")
        if not isinstance(thought_data["thought_number"], int):
            raise ValueError("'thought_number' must be an integer")
        if not isinstance(thought_data["total_thoughts"], int):
            raise ValueError("'total_thoughts' must be an integer")
        if not isinstance(thought_data["next_thought_needed"], bool):
            raise ValueError("'next_thought_needed' must be a boolean")
    
    def _prepare_context_for_llm(self) -> str:
        """Prepare a context string from all thoughts for the LLM."""
        if not self.thoughts:
            return "No previous thoughts."
        
        context_lines = []
        for thought in self.thoughts:
            prefix = ""
            if thought.get("is_revision"):
                prefix = f"[Revision of Thought {thought.get('revises_thought')}] "
            elif thought.get("branch_from_thought"):
                prefix = f"[Branch from Thought {thought.get('branch_from_thought')}] "
            
            context_lines.append(
                f"Thought {thought['thought_number']}/{thought['total_thoughts']}: "
                f"{prefix}{thought['thought']}"
            )
        
        return "\n\n".join(context_lines)
    
    def _extract_json_from_response(self, response: str) -> Dict[str, Any]:
        """Extract JSON object from LLM response."""
        # Look for JSON patterns in the response
        import re
        
        # Try to find a JSON block with or without code markers
        json_pattern = r'```(?:json)?\s*(.*?)```'
        match = re.search(json_pattern, response, re.DOTALL)
        
        if match:
            # Found JSON within code markers
            json_str = match.group(1).strip()
        else:
            # Try to find JSON without code markers - look for the whole response
            json_str = response.strip()
        
        try:
            # Parse the JSON
            thought_data = json.loads(json_str)
            
            # Convert field names to snake_case if they're in camelCase
            keys_mapping = {
                "thoughtNumber": "thought_number",
                "totalThoughts": "total_thoughts",
                "nextThoughtNeeded": "next_thought_needed",
                "isRevision": "is_revision",
                "revisesThought": "revises_thought",
                "branchFromThought": "branch_from_thought",
                "branchId": "branch_id",
                "needsMoreThoughts": "needs_more_thoughts"
            }
            
            # Convert keys
            for camel_case, snake_case in keys_mapping.items():
                if camel_case in thought_data:
                    thought_data[snake_case] = thought_data.pop(camel_case)
            
            return thought_data
            
        except json.JSONDecodeError:
            # Failed to parse JSON, raise error
            raise ValueError(f"Failed to parse JSON from response: {response}")
    
    async def _save_session_state(self) -> None:
        """Save the session state to Redis."""
        state_manager = get_state_manager()
        session_key = f"sequential_thinking:{self.session_id}"
        
        session_state = {
            "problem": self.problem,
            "thoughts": self.thoughts,
            "branches": self.branches,
            "current_branch_id": self.current_branch_id
        }
        
        await state_manager.set(session_key, session_state, ttl=3600)  # 1 hour TTL
    
    @classmethod
    async def load_session(cls, session_id: str) -> "SequentialThinkingSession":
        """
        Load a session from Redis.
        
        Args:
            session_id: Session ID
            
        Returns:
            Loaded session or None if not found
        """
        state_manager = get_state_manager()
        session_key = f"sequential_thinking:{session_id}"
        
        session_state = await state_manager.get(session_key)
        
        if not session_state:
            raise ValueError(f"Session not found: {session_id}")
        
        session = cls(session_id, session_state["problem"])
        session.thoughts = session_state["thoughts"]
        session.branches = session_state["branches"]
        session.current_branch_id = session_state["current_branch_id"]
        
        return session

@tool(name="sequential_thinking", description="Break down and solve problems through sequential thinking with LLM")
async def sequential_thinking(
    problem: str,
    thought: Optional[str] = None,
    session_id: Optional[str] = None,
    thought_number: Optional[int] = None,
    total_thoughts: Optional[int] = None,
    next_thought_needed: Optional[bool] = None,
    is_revision: Optional[bool] = None,
    revises_thought: Optional[int] = None,
    branch_from_thought: Optional[int] = None,
    branch_id: Optional[str] = None,
    needs_more_thoughts: Optional[bool] = None,
    generate_final_answer: bool = False
) -> Dict[str, Any]:
    """
    Advanced sequential thinking tool powered by LLM for step-by-step reasoning.
    
    Args:
        problem: The problem statement to analyze
        thought: Current thought content (if adding a specific thought)
        session_id: Session identifier for continuing previous thinking
        thought_number: Current thought number in sequence
        total_thoughts: Estimated total thoughts needed
        next_thought_needed: Whether another thought is needed
        is_revision: Whether this thought revises previous thinking
        revises_thought: Which thought is being reconsidered
        branch_from_thought: Thought number to branch from
        branch_id: Branch identifier
        needs_more_thoughts: Whether more thoughts are needed
        generate_final_answer: Whether to generate a final answer
        
    Returns:
        Sequential thinking results with thought information
    """
    # Create or load session
    if session_id:
        try:
            # Try to load existing session
            session = await SequentialThinkingSession.load_session(session_id)
        except ValueError:
            # Session not found, create new one
            session_id = f"st_{hash(problem) % 10000:04d}"
            session = SequentialThinkingSession(session_id, problem)
    else:
        # Create new session
        session_id = f"st_{hash(problem) % 10000:04d}"
        session = SequentialThinkingSession(session_id, problem)
    
    # Process based on inputs
    if thought:
        # Add specific thought provided by the caller
        thought_data = {
            "thought": thought,
            "thought_number": thought_number or (len(session.thoughts) + 1),
            "total_thoughts": total_thoughts or max(3, len(session.thoughts) + 2),
            "next_thought_needed": next_thought_needed if next_thought_needed is not None else True
        }
        
        # Add optional fields
        if is_revision:
            thought_data["is_revision"] = is_revision
        if revises_thought:
            thought_data["revises_thought"] = revises_thought
        if branch_from_thought:
            thought_data["branch_from_thought"] = branch_from_thought
        if branch_id:
            thought_data["branch_id"] = branch_id
        if needs_more_thoughts:
            thought_data["needs_more_thoughts"] = needs_more_thoughts
        
        result = await session.add_thought(thought_data)
    elif generate_final_answer:
        # Generate final answer based on the thought sequence
        final_answer = await session.get_final_answer()
        
        # Add a concluding thought
        thought_data = {
            "thought": f"Final conclusion: {final_answer[:100]}...",
            "thought_number": len(session.thoughts) + 1,
            "total_thoughts": len(session.thoughts) + 1,
            "next_thought_needed": False
        }
        
        await session.add_thought(thought_data)
        
        return {
            "session_id": session_id,
            "problem": problem,
            "final_answer": final_answer,
            "thoughts_count": len(session.thoughts),
            "thoughts": session.thoughts
        }
    else:
        # Generate the next thought automatically
        thought_data = await session.generate_next_thought()
        result = await session.add_thought(thought_data)
    
    # Prepare response
    response = {
        "session_id": session_id,
        "problem": problem,
        "current_thought": thought_data,
        "thought_history_length": len(session.thoughts),
        "branches": list(session.branches.keys()),
        "next_thought_needed": thought_data["next_thought_needed"]
    }
    
    # Add full thought history if it's small
    if len(session.thoughts) <= 10:
        response["thoughts"] = session.thoughts
    
    return response