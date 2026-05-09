"""Shared utilities for the Obsidian vault MCP server."""

import json
from datetime import date, datetime


def sanitize_for_json(obj):
    """Recursively convert date/datetime objects to ISO strings for JSON serialisation."""
    if isinstance(obj, dict):
        return {k: sanitize_for_json(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [sanitize_for_json(v) for v in obj]
    elif isinstance(obj, (date, datetime)):
        return obj.isoformat()
    return obj


class SafeJSONEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, (date, datetime)):
            return obj.isoformat()
        return super().default(obj)
