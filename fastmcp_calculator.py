#!/usr/bin/env python3
"""
FastMCP Calculator Server
=========================

A simple MCP server that provides basic mathematical calculation tools.
This demonstrates how to create MCP tools for computational tasks.

Requirements:
pip install fastmcp

Usage:
python fastmcp_calculator.py

Author: Your Name
Date: 2024
"""

from fastmcp import FastMCP
import math
import statistics
from typing import List, Union

# Initialize the FastMCP server
mcp = FastMCP("Calculator Server")

@mcp.tool()
def add_numbers(a: float, b: float) -> float:
    """
    Add two numbers together.
    
    Args:
        a: First number to add
        b: Second number to add
    
    Returns:
        The sum of a and b
    """
    result = a + b
    print(f"Adding {a} + {b} = {result}")
    return result

@mcp.tool()
def subtract_numbers(a: float, b: float) -> float:
    """
    Subtract the second number from the first.
    
    Args:
        a: Number to subtract from
        b: Number to subtract
    
    Returns:
        The difference of a - b
    """
    result = a - b
    print(f"Subtracting {a} - {b} = {result}")
    return result

@mcp.tool()
def multiply_numbers(a: float, b: float) -> float:
    """
    Multiply two numbers together.
    
    Args:
        a: First number to multiply
        b: Second number to multiply
    
    Returns:
        The product of a and b
    """
    result = a * b
    print(f"Multiplying {a} × {b} = {result}")
    return result

@mcp.tool()
def divide_numbers(a: float, b: float) -> float:
    """
    Divide the first number by the second.
    
    Args:
        a: Dividend (number to be divided)
        b: Divisor (number to divide by)
    
    Returns:
        The quotient of a / b
        
    Raises:
        ValueError: If attempting to divide by zero
    """
    if b == 0:
        raise ValueError("Cannot divide by zero!")
    
    result = a / b
    print(f"Dividing {a} ÷ {b} = {result}")
    return result

@mcp.tool()
def power_calculation(base: float, exponent: float) -> float:
    """
    Calculate base raised to the power of exponent.
    
    Args:
        base: The base number
        exponent: The power to raise the base to
    
    Returns:
        base ^ exponent
    """
    result = math.pow(base, exponent)
    print(f"Calculating {base}^{exponent} = {result}")
    return result

@mcp.tool()
def square_root(number: float) -> float:
    """
    Calculate the square root of a number.
    
    Args:
        number: Number to find the square root of
    
    Returns:
        Square root of the number
        
    Raises:
        ValueError: If number is negative
    """
    if number < 0:
        raise ValueError("Cannot calculate square root of negative number!")
    
    result = math.sqrt(number)
    print(f"Square root of {number} = {result}")
    return result

@mcp.tool()
def calculate_percentage(part: float, whole: float) -> float:
    """
    Calculate what percentage 'part' is of 'whole'.
    
    Args:
        part: The part value
        whole: The whole value
    
    Returns:
        Percentage value
        
    Raises:
        ValueError: If whole is zero
    """
    if whole == 0:
        raise ValueError("Cannot calculate percentage when whole is zero!")
    
    result = (part / whole) * 100
    print(f"{part} is {result:.2f}% of {whole}")
    return result

@mcp.tool()
def list_statistics(numbers: List[float]) -> dict:
    """
    Calculate basic statistics for a list of numbers.
    
    Args:
        numbers: List of numbers to analyze
    
    Returns:
        Dictionary containing mean, median, mode (if exists), min, max, and sum
        
    Raises:
        ValueError: If the list is empty
    """
    if not numbers:
        raise ValueError("Cannot calculate statistics for empty list!")
    
    try:
        mode_value = statistics.mode(numbers)
    except statistics.StatisticsError:
        mode_value = None  # No unique mode
    
    stats = {
        "count": len(numbers),
        "sum": sum(numbers),
        "mean": statistics.mean(numbers),
        "median": statistics.median(numbers),
        "mode": mode_value,
        "min": min(numbers),
        "max": max(numbers),
        "range": max(numbers) - min(numbers)
    }
    
    if len(numbers) > 1:
        stats["standard_deviation"] = statistics.stdev(numbers)
    else:
        stats["standard_deviation"] = 0
    
    print(f"Statistics for {numbers}:")
    for key, value in stats.items():
        if value is not None:
            if isinstance(value, float):
                print(f"  {key}: {value:.4f}")
            else:
                print(f"  {key}: {value}")
    
    return stats

@mcp.tool()
def compound_interest(principal: float, rate: float, time: float, 
                     compounds_per_year: int = 1) -> dict:
    """
    Calculate compound interest.
    
    Args:
        principal: Initial amount of money
        rate: Annual interest rate (as decimal, e.g., 0.05 for 5%)
        time: Number of years
        compounds_per_year: How many times interest compounds per year (default: 1)
    
    Returns:
        Dictionary with final amount, interest earned, and breakdown
    """
    if principal <= 0:
        raise ValueError("Principal must be positive!")
    if rate < 0:
        raise ValueError("Interest rate cannot be negative!")
    if time < 0:
        raise ValueError("Time cannot be negative!")
    if compounds_per_year <= 0:
        raise ValueError("Compounds per year must be positive!")
    
    # A = P(1 + r/n)^(nt)
    final_amount = principal * math.pow(1 + rate/compounds_per_year, 
                                       compounds_per_year * time)
    interest_earned = final_amount - principal
    
    result = {
        "principal": principal,
        "rate_percent": rate * 100,
        "time_years": time,
        "compounds_per_year": compounds_per_year,
        "final_amount": final_amount,
        "interest_earned": interest_earned,
        "total_return_percent": (interest_earned / principal) * 100
    }
    
    print(f"Compound Interest Calculation:")
    print(f"  Principal: ${principal:,.2f}")
    print(f"  Rate: {rate*100:.2f}% annually")
    print(f"  Time: {time} years")
    print(f"  Compounds: {compounds_per_year} times per year")
    print(f"  Final Amount: ${final_amount:,.2f}")
    print(f"  Interest Earned: ${interest_earned:,.2f}")
    print(f"  Total Return: {(interest_earned/principal)*100:.2f}%")
    
    return result

@mcp.tool()
def solve_quadratic(a: float, b: float, c: float) -> dict:
    """
    Solve quadratic equation ax² + bx + c = 0 using the quadratic formula.
    
    Args:
        a: Coefficient of x²
        b: Coefficient of x
        c: Constant term
    
    Returns:
        Dictionary with solutions and discriminant info
        
    Raises:
        ValueError: If 'a' is zero (not a quadratic equation)
    """
    if a == 0:
        raise ValueError("Coefficient 'a' cannot be zero for quadratic equation!")
    
    # Calculate discriminant
    discriminant = b**2 - 4*a*c
    
    result = {
        "equation": f"{a}x² + {b}x + {c} = 0",
        "discriminant": discriminant
    }
    
    if discriminant > 0:
        # Two real solutions
        x1 = (-b + math.sqrt(discriminant)) / (2*a)
        x2 = (-b - math.sqrt(discriminant)) / (2*a)
        result["solutions"] = [x1, x2]
        result["solution_type"] = "Two real solutions"
        
    elif discriminant == 0:
        # One real solution
        x = -b / (2*a)
        result["solutions"] = [x]
        result["solution_type"] = "One real solution (repeated root)"
        
    else:
        # Complex solutions
        real_part = -b / (2*a)
        imaginary_part = math.sqrt(abs(discriminant)) / (2*a)
        result["solutions"] = [
            f"{real_part:.4f} + {imaginary_part:.4f}i",
            f"{real_part:.4f} - {imaginary_part:.4f}i"
        ]
        result["solution_type"] = "Two complex solutions"
    
    print(f"Quadratic equation: {result['equation']}")
    print(f"Discriminant: {discriminant}")
    print(f"Solution type: {result['solution_type']}")
    print(f"Solutions: {result['solutions']}")
    
    return result

# Resource to provide calculation examples and documentation
@mcp.resource("calculator://examples")
def get_calculator_examples():
    """
    Provide examples of how to use the calculator tools.
    """
    return """
# Calculator MCP Server Examples

## Basic Operations
- add_numbers(10, 5) → 15
- subtract_numbers(10, 5) → 5  
- multiply_numbers(10, 5) → 50
- divide_numbers(10, 5) → 2.0

## Advanced Operations
- power_calculation(2, 8) → 256.0
- square_root(16) → 4.0
- calculate_percentage(25, 100) → 25.0

## Statistics
- list_statistics([1, 2, 3, 4, 5]) → {mean: 3.0, median: 3.0, ...}

## Financial Calculations
- compound_interest(1000, 0.05, 10) → Calculate 5% interest over 10 years

## Equation Solving
- solve_quadratic(1, -5, 6) → Solve x² - 5x + 6 = 0

All functions include proper error handling and detailed output formatting.
"""

if __name__ == "__main__":
    # Print server information
    print("🧮 FastMCP Calculator Server")
    print("=" * 40)
    print(f"Server Name: {mcp.name}")
    print("Available Tools:")
    
    # List all available tools
    tools = [
        "add_numbers", "subtract_numbers", "multiply_numbers", "divide_numbers",
        "power_calculation", "square_root", "calculate_percentage", 
        "list_statistics", "compound_interest", "solve_quadratic"
    ]
    
    for i, tool in enumerate(tools, 1):
        print(f"  {i:2d}. {tool}")
    
    print("\nStarting MCP server...")
    print("The server will handle calculation requests via MCP protocol.")
    print("Press Ctrl+C to stop the server.")
    print("=" * 40)
    
    # Run the server
    mcp.run()
