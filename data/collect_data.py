import os
import time
import json
from pathlib import Path
import requests
from dotenv import load_dotenv

load_dotenv()  # Loads variables from .env

db_url = os.getenv("DATABASE_URL")

auth = (os.getenv("RAVELRY_USER"), os.getenv("RAVELRY_PASS"))
BASE_URL = "https://api.ravelry.com"

# Pattern-category permalink to restrict results to. "sweater" is the
# parent category and includes cardigan / pullover / vest as children.
PATTERN_CATEGORY = "sweater"

# How many photos to take per pattern. They are spread across the
# pattern's full photo list so the dataset gets a mix of hero shots,
# flat-lays, and stitch-detail close-ups instead of N near-identical
# styled shots from the front of the list.
PHOTOS_PER_PATTERN = 3


def search_patterns(query, page=1, page_size=100, pattern_category=None):
    """Search for patterns by stitch type / keyword.

    pattern_category: optional Ravelry category permalink (e.g. "sweater")
    to restrict results to a garment type.
    """
    params = {
        "query": query,
        "page": page,
        "page_size": page_size,
        "photo": "yes",  # only return patterns that have photos
        "craft": "knitting",
    }
    if pattern_category:
        # "pc" is the pattern-category filter used by Ravelry's search.
        params["pc"] = pattern_category

    response = requests.get(
        f"{BASE_URL}/patterns/search.json",
        auth=auth,
        params=params,
    )
    return response.json()


def get_pattern_details(pattern_id):
    """Get full pattern details including photos and yarn weight."""
    response = requests.get(
        f"{BASE_URL}/patterns/{pattern_id}.json",
        auth=auth
    )
    return response.json().get("pattern", {})


def select_photos(photos, n):
    """Pick up to n photos spread evenly across the pattern's photo list.

    The first photo is usually the styled "hero" shot; later photos tend
    to include flat-lays and stitch-detail close-ups. Spreading the picks
    gives the training set a mix of both rather than n near-identical
    shots from the front of the list.
    """
    if not photos:
        return []
    if len(photos) <= n:
        return photos
    # Evenly spaced indices across the full list, e.g. for 12 photos and
    # n=3 this picks indices 0, 4, 8.
    step = len(photos) / n
    return [photos[int(i * step)] for i in range(n)]


STITCH_QUERIES = ["cable knit", "seed stitch", "stockinette", "ribbing", "lace", "colorwork"]
OUTPUT_DIR = Path("data/datasetnew")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)

metadata = []

MAX_IMAGES = 1500
MAX_PER_QUERY = MAX_IMAGES // len(STITCH_QUERIES)  

for query in STITCH_QUERIES:
    query_count = 0  # reset for each query

    results = search_patterns(query, page_size=100, pattern_category=PATTERN_CATEGORY)
    pattern_ids = [p["id"] for p in results.get("patterns", [])]

    for pattern_id in pattern_ids:
        if query_count >= MAX_PER_QUERY:
            break

        details = get_pattern_details(pattern_id)
        photos = details.get("photos", [])
        yarn_weight = details.get("yarn_weight", {})
        gauge_pattern = details.get("gauge_pattern", "")

        for photo in select_photos(photos, PHOTOS_PER_PATTERN):
            if query_count >= MAX_PER_QUERY:
                break

            img_url = photo.get("medium2_url") or photo.get("medium_url")
            if not img_url:
                continue

            img_data = requests.get(img_url).content
            filename = f"{pattern_id}_{photo['id']}.jpg"
            filepath = OUTPUT_DIR / "images" / filename

            with open(filepath, "wb") as f:
                f.write(img_data)

            # Build a text caption for training
            weight_name = yarn_weight.get("name", "") if yarn_weight else ""
            caption = f"{query}, {weight_name}, {gauge_pattern}".strip(", ")

            metadata.append({
                "filename": filename,
                "caption": caption,
                "pattern_id": pattern_id,
                "yarn_weight": weight_name,
                "stitch_type": query,
            })

            query_count += 1
            time.sleep(0.3)

    print(f"{query}: downloaded {query_count} images")


# Save captions as JSON
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Downloaded {len(metadata)} images")
