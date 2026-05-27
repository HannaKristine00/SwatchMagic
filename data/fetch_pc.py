"""
Fetch the Ravelry pattern-category tree and print it as an indented
hierarchy. Any category whose name contains "sweater" (or common
sweater sub-types) is highlighted so you can copy its exact `permalink`.

Usage:
    python inspect_categories.py

Requires RAVELRY_USER and RAVELRY_PASS in your .env file (same as collect_data.py).
"""

import os
import requests
from dotenv import load_dotenv

load_dotenv()

auth = (os.getenv("RAVELRY_USER"), os.getenv("RAVELRY_PASS"))
BASE_URL = "https://api.ravelry.com"

# Words that suggest a category is sweater-related. Adjust as you like.
SWEATER_HINTS = ["sweater", "pullover", "cardigan", "vest", "jumper"]


def get_pattern_categories():
    """Return the full nested pattern-category tree."""
    response = requests.get(
        f"{BASE_URL}/pattern_categories/list.json",
        auth=auth,
    )
    response.raise_for_status()
    return response.json().get("pattern_categories", {})


def is_sweatery(name):
    lowered = name.lower()
    return any(hint in lowered for hint in SWEATER_HINTS)


def walk(node, depth=0):
    """Recursively print a category and its children."""
    name = node.get("name", "")
    permalink = node.get("permalink", "")
    indent = "  " * depth

    marker = "  <-- SWEATER" if is_sweatery(name) else ""
    print(f"{indent}{name}  [permalink: {permalink}]{marker}")

    for child in node.get("children", []):
        walk(child, depth + 1)


def collect_sweater_permalinks(node, found):
    """Recursively gather permalinks for sweater-related categories."""
    if is_sweatery(node.get("name", "")):
        found.append((node.get("name", ""), node.get("permalink", "")))
    for child in node.get("children", []):
        collect_sweater_permalinks(child, found)


if __name__ == "__main__":
    categories = get_pattern_categories()

    # The API returns a top-level dict; its children are the real tree.
    roots = categories.get("children", [])

    print("=" * 60)
    print("FULL PATTERN CATEGORY TREE")
    print("=" * 60)
    for root in roots:
        walk(root)

    print()
    print("=" * 60)
    print("SWEATER-RELATED CATEGORIES (copy these permalinks)")
    print("=" * 60)
    found = []
    for root in roots:
        collect_sweater_permalinks(root, found)

    if found:
        for name, permalink in found:
            print(f"  {name:30s} -> {permalink}")
    else:
        print("  None found -- check the printed tree above for the right name.")