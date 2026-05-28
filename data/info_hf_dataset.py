import random
from datasets import load_from_disk
import matplotlib.pyplot as plt
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
ds = load_from_disk(BASE_DIR / "data" / "dataset" / "hf_dataset")

# Pick 12 random indices
indices = random.sample(range(len(ds)), 12)

fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i, idx in enumerate(indices):
    sample = ds[idx]
    axes[i].imshow(sample["image"])
    axes[i].set_title(sample["text"], fontsize=7, wrap=True)
    axes[i].axis("off")

plt.suptitle("Random samples from training dataset", fontsize=12)
plt.tight_layout()
# plt.savefig("dataset_preview.png", dpi=150)
plt.show()