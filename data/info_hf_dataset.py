from datasets import load_from_disk
import matplotlib.pyplot as plt

# Load the dataset
ds = load_from_disk("data/dataset/hf_dataset")

# Basic info
print(ds)
print(f"\nNumber of examples: {len(ds)}")
print(f"Features: {ds.features}")
print(f"\nExample captions:")
for i in range(5):
    print(f"  [{i}] {ds[i]['text']}")

# Visualize a grid of sample images
fig, axes = plt.subplots(3, 4, figsize=(12, 9))
axes = axes.flatten()

for i, ax in enumerate(axes):
    sample = ds[i]
    ax.imshow(sample["image"])
    ax.set_title(sample["text"], fontsize=7, wrap=True)
    ax.axis("off")

plt.suptitle("Sample images from training dataset", fontsize=12)
plt.tight_layout()
#plt.savefig("dataset_preview.png", dpi=150)
plt.show()