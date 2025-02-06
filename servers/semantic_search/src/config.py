import os
from typing import Optional, List
from dataclasses import dataclass


@dataclass
class SearchConfig:
    """Configuration for semantic search settings."""

    # Required settings
    qdrant_url: str = os.getenv("QDRANT_URL", "http://localhost:6333")
    collection_name: str = os.getenv("QDRANT_COLLECTION_NAME", "neuroconv")
    qdrant_api_key: Optional[str] = os.getenv("QDRANT_API_KEY", None)

    # Optional settings with defaults
    timeout: float = 60.0
    return_digest_summary: bool = True
    return_references: bool = True
    limit: int = 10
    model: str = "openai/o3-mini"
    keywords: Optional[List[str]] = None


# Default configuration instance
config = SearchConfig()
