from datasets import Dataset
from PIL import Image
import json

with open("dataset/metadata.json") as f:
    metadata = json.load(f)

def load_example(item):
    img = Image.open(f"dataset/images/{item['filename']}").convert("RGB")
    return {"image": img, "text": item["caption"]}

examples = [load_example(m) for m in metadata]
ds = Dataset.from_list(examples)
ds.save_to_disk("dataset/hf_dataset")
print(ds)