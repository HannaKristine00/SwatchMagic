"""
SwatchMagic — LoRA Checkpoint Evaluation
=========================================
Run after each training checkpoint to measure:
  1. Image generation (seen + unseen prompts)
  2. FID against validation split
  3. Domain classifier accuracy (stitch type + yarn weight)
  4. CLIP score (prompt alignment)
  5. SSIM/LPIPS memorization check
  6. Diversity score (intra-prompt LPIPS variance)

Usage:
    python evaluate_checkpoint.py \
        --checkpoint  training/lora_weights/checkpoint_epoch5 \
        --val_dataset data/dataset/hf_dataset_val \
        --train_meta  data/dataset/metadata.json \
        --output_dir  evaluation/epoch5
"""

import argparse
import json
import csv
import os
import random
from pathlib import Path
from datetime import datetime

import torch
import numpy as np
from PIL import Image
from tqdm import tqdm

# Diffusers / transformers
from diffusers import StableDiffusionPipeline
from transformers import (
    CLIPProcessor, CLIPModel,
    ViTForImageClassification, ViTFeatureExtractor,
    AutoFeatureExtractor,
)

# Torchvision / metrics
from torchvision import transforms
from torchvision.models import inception_v3

# pip install torch-fidelity lpips
import lpips
from skimage.metrics import structural_similarity as ssim_fn

# ---------------------------------------------------------------------------
# Prompt sets
# ---------------------------------------------------------------------------

SEEN_PROMPTS = [
    "cable knit swatch, bulky yarn",
    "seed stitch swatch, fingering weight yarn",
    "stockinette swatch, worsted weight yarn",
    "ribbing swatch, DK weight yarn",
    "lace swatch, lace weight yarn",
    "colorwork swatch, sport weight yarn",
]

UNSEEN_PROMPTS = [
    "brioche stitch swatch, chunky yarn",
    "moss stitch swatch, aran weight yarn",
    "basketweave swatch, bulky yarn",
    "honeycomb stitch swatch, fingering weight yarn",
    "twisted rib swatch, DK weight yarn",
    "double knitting swatch, worsted weight yarn",
]

# Stitch and yarn weight label maps — must match your classifier training labels
STITCH_LABELS  = ["cable knit", "seed stitch", "stockinette", "ribbing", "lace", "colorwork"]
WEIGHT_LABELS  = ["lace", "fingering", "sport", "DK", "worsted", "aran", "bulky"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_pipeline(model_id: str, checkpoint_dir: str, device: str):
    pipe = StableDiffusionPipeline.from_pretrained(
        model_id,
        torch_dtype=torch.float16 if device == "cuda" else torch.float32,
    )
    pipe.unet.load_attn_procs(checkpoint_dir)
    pipe = pipe.to(device)
    pipe.safety_checker = None          # disable for swatch images
    pipe.set_progress_bar_config(disable=True)
    return pipe


def generate_images(pipe, prompts: list[str], n_per_prompt: int, device: str) -> list[dict]:
    """Generate n_per_prompt images for each prompt. Returns list of {image, prompt}."""
    results = []
    for prompt in tqdm(prompts, desc="Generating images"):
        for _ in range(n_per_prompt):
            with torch.autocast(device):
                out = pipe(prompt, num_inference_steps=30, guidance_scale=7.5)
            results.append({"image": out.images[0], "prompt": prompt})
    return results


def pil_to_tensor(img: Image.Image, size=299) -> torch.Tensor:
    t = transforms.Compose([
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])
    return t(img.convert("RGB"))


# ---------------------------------------------------------------------------
# 1 + 2  FID
# ---------------------------------------------------------------------------

def compute_fid(generated_images: list[Image.Image], val_images: list[Image.Image], device: str) -> float:
    """
    Compute FID using Inception-v3 feature vectors.
    For a cleaner implementation you can also use `torch-fidelity`:
        torchmetrics.image.fid.FrechetInceptionDistance
    This version is self-contained.
    """
    from torchmetrics.image.fid import FrechetInceptionDistance

    fid_metric = FrechetInceptionDistance(feature=2048).to(device)

    def batch_to_uint8(imgs):
        tensors = []
        for img in imgs:
            t = transforms.Compose([
                transforms.Resize((299, 299)),
                transforms.ToTensor(),
            ])(img.convert("RGB"))
            tensors.append((t * 255).to(torch.uint8))
        return torch.stack(tensors).to(device)

    print("  Computing FID — loading real images...")
    fid_metric.update(batch_to_uint8(val_images), real=True)

    print("  Computing FID — loading generated images...")
    fid_metric.update(batch_to_uint8(generated_images), real=False)

    score = fid_metric.compute().item()
    return score


# ---------------------------------------------------------------------------
# 3  Domain classifier
# ---------------------------------------------------------------------------

class DomainClassifier:
    """
    Thin wrapper around two fine-tuned ViT classifiers:
      - stitch type  (6 classes)
      - yarn weight  (7 classes)

    If you haven't trained these yet, see train_classifier.py (to be added).
    Pass None to skip that head.
    """

    def __init__(self, stitch_ckpt: str | None, weight_ckpt: str | None, device: str):
        self.device = device
        self.stitch_model = None
        self.weight_model = None

        self.feature_extractor = AutoFeatureExtractor.from_pretrained("google/vit-base-patch16-224")

        if stitch_ckpt and Path(stitch_ckpt).exists():
            self.stitch_model = ViTForImageClassification.from_pretrained(
                stitch_ckpt, num_labels=len(STITCH_LABELS), ignore_mismatched_sizes=True
            ).to(device).eval()

        if weight_ckpt and Path(weight_ckpt).exists():
            self.weight_model = ViTForImageClassification.from_pretrained(
                weight_ckpt, num_labels=len(WEIGHT_LABELS), ignore_mismatched_sizes=True
            ).to(device).eval()

    def predict(self, image: Image.Image):
        inputs = self.feature_extractor(images=image, return_tensors="pt").to(self.device)
        stitch_pred = weight_pred = None

        with torch.no_grad():
            if self.stitch_model:
                logits = self.stitch_model(**inputs).logits
                stitch_pred = STITCH_LABELS[logits.argmax(-1).item()]
            if self.weight_model:
                logits = self.weight_model(**inputs).logits
                weight_pred = WEIGHT_LABELS[logits.argmax(-1).item()]

        return stitch_pred, weight_pred


def run_classifier_eval(classifier: DomainClassifier, generated: list[dict]) -> dict:
    """
    Compute top-1 accuracy for stitch type and yarn weight.
    Ground truth is inferred from the prompt text.
    """
    stitch_correct = stitch_total = 0
    weight_correct = weight_total = 0

    for item in tqdm(generated, desc="  Classifier eval"):
        prompt = item["prompt"].lower()
        stitch_pred, weight_pred = classifier.predict(item["image"])

        # Ground-truth from prompt
        gt_stitch = next((s for s in STITCH_LABELS if s in prompt), None)
        gt_weight  = next((w for w in WEIGHT_LABELS if w in prompt), None)

        if gt_stitch and stitch_pred:
            stitch_correct += int(stitch_pred == gt_stitch)
            stitch_total   += 1

        if gt_weight and weight_pred:
            weight_correct += int(weight_pred == gt_weight)
            weight_total   += 1

    return {
        "stitch_accuracy": stitch_correct / stitch_total if stitch_total else None,
        "weight_accuracy": weight_correct / weight_total if weight_total else None,
    }


# ---------------------------------------------------------------------------
# 4  CLIP score
# ---------------------------------------------------------------------------

def compute_clip_scores(generated: list[dict], device: str) -> float:
    model     = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device).eval()
    processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

    scores = []
    for item in tqdm(generated, desc="  CLIP score"):
        inputs = processor(
            text=[item["prompt"]],
            images=item["image"],
            return_tensors="pt",
            padding=True,
        ).to(device)

        with torch.no_grad():
            outputs   = model(**inputs)
            img_emb   = outputs.image_embeds / outputs.image_embeds.norm(dim=-1, keepdim=True)
            txt_emb   = outputs.text_embeds  / outputs.text_embeds.norm(dim=-1, keepdim=True)
            score     = (img_emb * txt_emb).sum().item()

        scores.append(score)

    return float(np.mean(scores))


# ---------------------------------------------------------------------------
# 5  Memorization check (SSIM + LPIPS vs nearest training neighbour)
# ---------------------------------------------------------------------------

def compute_memorization(
    generated: list[dict],
    train_images: list[Image.Image],
    lpips_fn,
    device: str,
    sample_train: int = 100,
) -> dict:
    """
    For each generated image find the nearest training image by LPIPS,
    then also record SSIM. High similarity → memorization.
    We sample `sample_train` training images to keep runtime reasonable.
    """
    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Pre-compute training tensors on a random sample
    sample = random.sample(train_images, min(sample_train, len(train_images)))
    train_tensors = torch.stack([to_tensor(img.convert("RGB")) for img in sample]).to(device)

    min_lpips_scores = []
    max_ssim_scores  = []

    for item in tqdm(generated, desc="  Memorization check"):
        gen_t = to_tensor(item["image"].convert("RGB")).unsqueeze(0).to(device)

        # LPIPS: lower = more similar
        with torch.no_grad():
            dists = [lpips_fn(gen_t, train_tensors[i].unsqueeze(0)).item()
                     for i in range(len(train_tensors))]
        min_lpips = min(dists)
        nearest_idx = int(np.argmin(dists))

        # SSIM on nearest neighbour
        gen_np  = np.array(item["image"].convert("RGB").resize((256, 256)))
        near_np = np.array(sample[nearest_idx].convert("RGB").resize((256, 256)))
        ssim_score = ssim_fn(gen_np, near_np, channel_axis=-1, data_range=255)

        min_lpips_scores.append(min_lpips)
        max_ssim_scores.append(ssim_score)

    return {
        "mean_min_lpips":   float(np.mean(min_lpips_scores)),   # lower = more memorized
        "mean_max_ssim":    float(np.mean(max_ssim_scores)),    # higher = more memorized
        "pct_lpips_lt_0_1": float(np.mean([s < 0.1 for s in min_lpips_scores])),  # % suspiciously similar
    }


# ---------------------------------------------------------------------------
# 6  Diversity score (intra-prompt LPIPS variance)
# ---------------------------------------------------------------------------

def compute_diversity(generated: list[dict], lpips_fn, device: str) -> float:
    """
    For each prompt, compute mean pairwise LPIPS between all generated images.
    Returns the mean across all prompts. Higher = more diverse.
    """
    to_tensor = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]),
    ])

    # Group by prompt
    by_prompt: dict[str, list] = {}
    for item in generated:
        by_prompt.setdefault(item["prompt"], []).append(item["image"])

    prompt_diversities = []
    for prompt, images in tqdm(by_prompt.items(), desc="  Diversity score"):
        if len(images) < 2:
            continue
        tensors = [to_tensor(img.convert("RGB")).unsqueeze(0).to(device) for img in images]
        pairwise = []
        for i in range(len(tensors)):
            for j in range(i + 1, len(tensors)):
                with torch.no_grad():
                    d = lpips_fn(tensors[i], tensors[j]).item()
                pairwise.append(d)
        prompt_diversities.append(float(np.mean(pairwise)))

    return float(np.mean(prompt_diversities)) if prompt_diversities else 0.0


# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------

def save_results(results: dict, output_dir: Path):
    output_dir.mkdir(parents=True, exist_ok=True)

    # JSON — full record
    with open(output_dir / "metrics.json", "w") as f:
        json.dump(results, f, indent=2)

    # CSV — append row for easy cross-checkpoint comparison
    csv_path = output_dir.parent / "eval_summary.csv"
    file_exists = csv_path.exists()
    with open(csv_path, "a", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=results.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(results)

    print(f"\n  Results saved to {output_dir / 'metrics.json'}")
    print(f"  Summary row appended to {csv_path}")


def save_sample_grid(generated: list[dict], output_dir: Path, n: int = 16):
    """Save a quick visual grid of generated images."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sample = random.sample(generated, min(n, len(generated)))
    cols = 4
    rows = (len(sample) + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 3, rows * 3))
    axes = axes.flatten()

    for i, item in enumerate(sample):
        axes[i].imshow(item["image"])
        axes[i].set_title(item["prompt"][:40], fontsize=6, wrap=True)
        axes[i].axis("off")
    for j in range(i + 1, len(axes)):
        axes[j].axis("off")

    plt.tight_layout()
    plt.savefig(output_dir / "sample_grid.png", dpi=120)
    plt.close()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint",       required=True,  help="Path to LoRA checkpoint dir")
    parser.add_argument("--val_dataset",      required=True,  help="Path to HF val dataset on disk")
    parser.add_argument("--train_meta",       required=True,  help="Path to training metadata.json")
    parser.add_argument("--output_dir",       required=True,  help="Where to save eval results")
    parser.add_argument("--model_id",         default="runwayml/stable-diffusion-v1-5")
    parser.add_argument("--n_per_prompt",     type=int, default=20,
                        help="Images per prompt (20 seen + 20 unseen = 480 total with 12 prompts)")
    parser.add_argument("--stitch_classifier", default=None,  help="Path to stitch classifier checkpoint")
    parser.add_argument("--weight_classifier", default=None,  help="Path to yarn weight classifier checkpoint")
    parser.add_argument("--epoch",            type=int, default=0, help="Epoch number for logging")
    args = parser.parse_args()

    DEVICE = "cuda" if torch.cuda.is_available() else "mps" if torch.backends.mps.is_available() else "cpu"
    print(f"Device: {DEVICE}")

    output_dir = Path(args.output_dir)

    # ── Load pipeline ────────────────────────────────────────────────────────
    print("\n[1/6] Loading pipeline and generating images...")
    pipe = load_pipeline(args.model_id, args.checkpoint, DEVICE)

    all_prompts = SEEN_PROMPTS + UNSEEN_PROMPTS
    generated   = generate_images(pipe, all_prompts, args.n_per_prompt, DEVICE)
    print(f"  Generated {len(generated)} images ({len(SEEN_PROMPTS)} seen + {len(UNSEEN_PROMPTS)} unseen prompts)")

    gen_images = [g["image"] for g in generated]
    save_sample_grid(generated, output_dir)

    # ── Load val + train images ───────────────────────────────────────────────
    from datasets import load_from_disk
    val_ds     = load_from_disk(args.val_dataset)
    val_images = [val_ds[i]["image"] for i in range(len(val_ds))]

    with open(args.train_meta) as f:
        train_meta = json.load(f)
    train_dir   = Path(args.train_meta).parent / "images"
    train_images = []
    for m in train_meta:
        p = train_dir / m["filename"]
        if p.exists():
            train_images.append(Image.open(p).convert("RGB"))

    print(f"  Val images: {len(val_images)} | Train images: {len(train_images)}")

    # ── FID ──────────────────────────────────────────────────────────────────
    print("\n[2/6] Computing FID...")
    fid_score = compute_fid(gen_images, val_images, DEVICE)
    print(f"  FID: {fid_score:.2f}  (lower is better)")

    # ── Domain classifier ────────────────────────────────────────────────────
    print("\n[3/6] Running domain classifier...")
    classifier    = DomainClassifier(args.stitch_classifier, args.weight_classifier, DEVICE)
    clf_results   = run_classifier_eval(classifier, generated)
    print(f"  Stitch accuracy: {clf_results['stitch_accuracy']}")
    print(f"  Weight accuracy: {clf_results['weight_accuracy']}")

    # ── CLIP score ────────────────────────────────────────────────────────────
    print("\n[4/6] Computing CLIP score...")
    clip_score = compute_clip_scores(generated, DEVICE)
    print(f"  Mean CLIP cosine similarity: {clip_score:.4f}  (higher is better)")

    # ── Memorization check ────────────────────────────────────────────────────
    print("\n[5/6] Memorization check (LPIPS + SSIM vs train set)...")
    lpips_fn = lpips.LPIPS(net="alex").to(DEVICE)
    mem_results = compute_memorization(generated, train_images, lpips_fn, DEVICE)
    print(f"  Mean min-LPIPS to train: {mem_results['mean_min_lpips']:.4f}  (higher = less memorized)")
    print(f"  Mean max-SSIM to train:  {mem_results['mean_max_ssim']:.4f}   (lower = less memorized)")
    print(f"  % images w/ LPIPS < 0.1: {mem_results['pct_lpips_lt_0_1']*100:.1f}%  (ideally near 0%)")

    # ── Diversity score ───────────────────────────────────────────────────────
    print("\n[6/6] Computing diversity score...")
    diversity = compute_diversity(generated, lpips_fn, DEVICE)
    print(f"  Mean intra-prompt LPIPS: {diversity:.4f}  (higher = more diverse)")

    # ── Save ──────────────────────────────────────────────────────────────────
    results = {
        "timestamp":           datetime.now().isoformat(),
        "epoch":               args.epoch,
        "checkpoint":          args.checkpoint,
        "n_generated":         len(generated),
        "fid":                 round(fid_score, 3),
        "clip_score":          round(clip_score, 4),
        "stitch_accuracy":     clf_results["stitch_accuracy"],
        "weight_accuracy":     clf_results["weight_accuracy"],
        "mean_min_lpips":      round(mem_results["mean_min_lpips"], 4),
        "mean_max_ssim":       round(mem_results["mean_max_ssim"], 4),
        "pct_memorized":       round(mem_results["pct_lpips_lt_0_1"], 4),
        "diversity_lpips":     round(diversity, 4),
    }

    save_results(results, output_dir)

    print("\n── Summary ─────────────────────────────────────────────────")
    for k, v in results.items():
        print(f"  {k:<25} {v}")
    print("────────────────────────────────────────────────────────────\n")


if __name__ == "__main__":
    main()