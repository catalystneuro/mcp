#!/usr/bin/env python3
import logging
from datetime import datetime
from typing import List

from mcp.server.fastmcp import FastMCP, Context

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastMCP server
mcp = FastMCP("Example Server")


# Static resource example
@mcp.resource("example://static/greeting")
def get_static_greeting() -> str:
    """A simple static greeting resource"""
    return "Hello from the example server!"


# Dynamic resource example with parameter
@mcp.resource("example://{name}/greeting")
def get_dynamic_greeting(name: str) -> str:
    """A personalized greeting using a name parameter"""
    return f"Hello, {name}! Welcome to the example server."


# Tool example with progress reporting
@mcp.tool()
async def process_items(items: List[str], ctx: Context) -> str:
    """Process a list of items with progress updates

    Args:
        items: List of items to process
        ctx: MCP context for progress reporting

    Returns:
        Completion message
    """
    for i, item in enumerate(items):
        # Log progress
        ctx.info(f"Processing item {i+1}/{len(items)}: {item}")

        # Update progress percentage
        await ctx.report_progress(i, len(items))

    return "Successfully processed all items!"


# Tool example that returns current time
@mcp.tool()
def get_current_time(format: str = "24h") -> str:
    """Get the current time in specified format

    Args:
        format: Time format - either "12h" or "24h"

    Returns:
        Current time as string
    """
    now = datetime.now()
    if format == "12h":
        return now.strftime("%I:%M %p")
    return now.strftime("%H:%M")


# Prompt example
@mcp.prompt()
def help_prompt() -> str:
    """Help template for using the server"""
    return """
    Welcome to the Example Server!

    Available features:
    - Static greeting: Access a simple greeting
    - Dynamic greeting: Get a personalized greeting with your name
    - Process items: Process a list of items with progress tracking
    - Get time: Get the current time in 12h or 24h format

    How can I help you use these features?
    """


if __name__ == "__main__":
    logger.info("Starting the server")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    logger.info("Exiting the server")
