#!/usr/bin/env python3
import asyncio
from time import sleep
from mcp.server.fastmcp import FastMCP

from search import search
from config import config


mcp = FastMCP(
  "semantic-search-server",
  version="0.1.0",
  request_timeout=300,
)


@mcp.tool()
def return_string(query: str, context: str) -> str:
    """
    Return a string for testing.

    Args:
      query: The search query text.
      context: The relevant context for the query.

    Returns:
      The string representation of the search result.
    """
    sleep(10)
    return f"Query: {query}, Context: {context}"


@mcp.tool()
async def search_about_neuroconv(query: str, context: str) -> str:
    """
    Search to learn about NeuroConv.

    Args:
      query: The search query text.
      context: The relevant context for the query.

    Returns:
      The string representation of the search result.
    """
    result = await search(
        query=query,
        context=context,
        qdrant_url=config.qdrant_url,
        collection_name=config.collection_name,
        keywords=config.keywords,
        qdrant_api_key=config.qdrant_api_key,
        timeout=config.timeout,
        return_digest_summary=config.return_digest_summary,
        return_references=config.return_references,
        limit=config.limit,
        model=config.model,
    )
    return str(result)
