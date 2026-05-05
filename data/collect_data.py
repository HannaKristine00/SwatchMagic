import os
import time
from pathlib import Path
import requests
from .env import RAVELRY_USER, RAVELRY_PASS

auth = (RAVELRY_USER, RAVELRY_PASS)
BASE_URL = "https://api.ravelry.com"

def search_patterns(query, page=1, page_size=100):
    """Search for patterns by stitch type / keyword."""
    response = requests.get(
        f"{BASE_URL}/patterns/search.json",
        auth=auth,
        params={
            "query": query,
            "page": page,
            "page_size": page_size,
            "photo": "yes",  # only return patterns that have photos
            "craft": "knitting",
        }
    )
    return response.json()

def get_pattern_details(pattern_id):
    """Get full pattern details including photos and yarn weight."""
    response = requests.get(
        f"{BASE_URL}/patterns/{pattern_id}.json",
        auth=auth
    )
    return response.json().get("pattern", {})