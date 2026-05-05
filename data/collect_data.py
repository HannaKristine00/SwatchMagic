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



STITCH_QUERIES = ["cable knit", "seed stitch", "stockinette", "ribbing", "lace", "colorwork"]
OUTPUT_DIR = Path("data/dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)

metadata = []

MAX_IMAGES = 500
MAX_PER_QUERY = MAX_IMAGES // len(STITCH_QUERIES)  # ~83 per query

for query in STITCH_QUERIES:
    query_count = 0  # reset for each query

    results = search_patterns(query, page_size=100)
    pattern_ids = [p["id"] for p in results.get("patterns", [])]

    for pattern_id in pattern_ids:
        if query_count >= MAX_PER_QUERY:
            break

        details = get_pattern_details(pattern_id)
        photos = details.get("photos", [])
        yarn_weight = details.get("yarn_weight", {})
        gauge_pattern = details.get("gauge_pattern", "")

        for photo in photos[:2]:
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