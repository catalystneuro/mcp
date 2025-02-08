#!/usr/bin/env python3
import logging
import asyncio
from time import sleep
from mcp.server.fastmcp import FastMCP

from search import search
from config import config


# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
    Use this tool to ask questions and learn about NeuroConv.
    NeuroConv is a Python package for converting neurophysiology data in a variety of proprietary formats to the Neurodata Without Borders (NWB) standard.
    Features:
    - Reads data from 40 popular neurophysiology data formats and writes to NWB using best practices.
    - Extracts relevant metadata from each format.
    - Handles large data volume by reading datasets piece-wise.
    - Minimizes the size of the NWB files by automatically applying chunking and lossless compression.
    - Supports ensembles of multiple data streams, and supports common methods for temporal alignment of streams.

    How to use search_about_neuroconv: Formulate a concise search query and provide the relevant context.
    Example:
    query = "Convert spiking data from Blackrock systems to NWB."
    context = "User wants to convert its recorded data to the NWB format, using NeuroConv."

    Args:
      query: A concise search query, containig the keywords of interest.
      context: A short but relevant context for the query.

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


if __name__ == "__main__":
    logger.info("Starting the server")
    try:
        mcp.run()
    except Exception as e:
        logger.error(f"An error occurred: {e}")
    logger.info("Exiting the server")

