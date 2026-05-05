import json

STITCH_QUERIES = ["cable knit", "seed stitch", "stockinette", "ribbing", "lace", "colorwork"]
OUTPUT_DIR = Path("data/dataset")
OUTPUT_DIR.mkdir(exist_ok=True)
(OUTPUT_DIR / "images").mkdir(exist_ok=True)

metadata = []

for query in STITCH_QUERIES:
    print(f"Searching: {query}")
    results = search_patterns(query, page_size=100)
    pattern_ids = [p["id"] for p in results.get("patterns", [])]

    for pattern_id in pattern_ids:
        details = get_pattern_details(pattern_id)
        photos = details.get("photos", [])
        yarn_weight = details.get("yarn_weight", {})
        gauge_pattern = details.get("gauge_pattern", "")  # e.g. "stockinette stitch"

        for photo in photos[:2]:  # max 2 photos per pattern
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

            time.sleep(0.3)  # be polite to the API

# Save captions as JSON
with open(OUTPUT_DIR / "metadata.json", "w") as f:
    json.dump(metadata, f, indent=2)

print(f"Downloaded {len(metadata)} images")