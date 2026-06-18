import os
import logging
from typing import Any, Dict

from reachy_mini_conversation_app.tools.core_tools import Tool, ToolDependencies


logger = logging.getLogger(__name__)

try:
    from firecrawl import Firecrawl

    FIRECRAWL_AVAILABLE = True
except ImportError:
    FIRECRAWL_AVAILABLE = False
    logger.warning("firecrawl-py not installed. Run: pip install firecrawl-py")


class SearchOnline(Tool):
    """Search the web using Firecrawl."""

    name = "search_online"
    description = (
        "Search the internet for up-to-date information on any topic. "
        "Returns a concise summary of the top web results. "
        "Use this when the user asks about current events, facts, news, or anything that requires live web data."
    )
    parameters_schema = {
        "type": "object",
        "properties": {
            "query": {
                "type": "string",
                "description": "The search query. Be specific. Use 'site:example.com' to restrict to a domain.",
            },
            "limit": {
                "type": "integer",
                "description": "Number of results to return (default: 5, max: 10).",
                "default": 5,
            },
        },
        "required": ["query"],
    }

    async def __call__(self, deps: ToolDependencies, **kwargs: Any) -> Dict[str, Any]:
        """Search the web with Firecrawl and return a summary of top results."""
        if not FIRECRAWL_AVAILABLE:
            return {"error": "firecrawl-py is not installed. Run: pip install firecrawl-py"}

        api_key = os.environ.get("FIRECRAWL_API_KEY")
        if not api_key:
            return {"error": "FIRECRAWL_API_KEY environment variable is not set."}

        query = kwargs.get("query", "").strip()
        if not query:
            return {"error": "A search query is required."}

        limit = min(int(kwargs.get("limit", 5)), 10)

        logger.info("Tool call: search_online query=%r limit=%d", query, limit)

        try:
            client = Firecrawl(api_key=api_key)
            results = client.search(query, limit=limit)

            hits = results.web or []
            if not hits:
                return {"query": query, "results": [], "summary": "No results found."}

            formatted = []
            for item in hits:
                formatted.append(
                    {
                        "title": getattr(item, "title", None),
                        "url": getattr(item, "url", None),
                        "description": getattr(item, "description", None),
                    }
                )

            # Build a short summary for the LLM to speak
            summary_lines = []
            for i, r in enumerate(formatted, 1):
                title = r.get("title") or "Untitled"
                desc = r.get("description") or ""
                url = r.get("url") or ""
                summary_lines.append(f"{i}. {title}: {desc} ({url})")

            summary = "\n".join(summary_lines)

            return {
                "query": query,
                "result_count": len(formatted),
                "results": formatted,
                "summary": summary,
            }

        except Exception as e:
            logger.exception("search_online failed")
            return {"error": f"Search failed: {e!s}"}
