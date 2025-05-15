from app.core.registry import tool
import asyncio
import random
import math

@tool(name="calculate", description="Perform a calculation", category="demo")
async def calculate(
    operation: str,
    a: float,
    b: float
) -> dict:
    """
    Perform a basic calculation.
    
    Args:
        operation: Type of operation (add, subtract, multiply, divide)
        a: First number
        b: Second number
        
    Returns:
        Result of the calculation
    """
    # Simulate processing time
    await asyncio.sleep(0.3)
    
    result = None
    if operation.lower() == "add":
        result = a + b
    elif operation.lower() == "subtract":
        result = a - b
    elif operation.lower() == "multiply":
        result = a * b
    elif operation.lower() == "divide":
        if b == 0:
            return {"error": "Cannot divide by zero"}
        result = a / b
    else:
        return {"error": f"Unknown operation: {operation}"}
    
    return {
        "operation": operation,
        "a": a,
        "b": b,
        "result": result
    }

@tool(name="random_number", description="Generate a random number within a range", category="demo")
async def random_number(
    min_value: int = 1,
    max_value: int = 100,
    count: int = 1
) -> dict:
    """
    Generate random numbers within a specified range.
    
    Args:
        min_value: Minimum value (inclusive)
        max_value: Maximum value (inclusive)
        count: Number of random numbers to generate
        
    Returns:
        Generated random numbers
    """
    # Input validation with O(1) complexity
    if min_value > max_value:
        min_value, max_value = max_value, min_value
    
    # Limit count for performance (O(n) operation)
    count = min(max(1, count), 1000)
    
    # Generate random numbers - O(n) complexity
    numbers = [random.randint(min_value, max_value) for _ in range(count)]
    
    return {
        "min": min_value,
        "max": max_value,
        "count": count,
        "numbers": numbers
    }