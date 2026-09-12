"""
generate_embeddings.py

Edit the CONFIG block below, then run:
    python generate_embeddings.py
"""

import json
import logging
import numpy as np
from pathlib import Path
from datetime import datetime

import torch
import clip
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

# ── CONFIG ────────────────────────────────────────────────────────────────────

HARMFUL_INPUT_DIR = "../images/..."
SAFE_INPUT_DIR    = "../images/..."


MODEL      = "ViT-L/14"
BATCH_SIZE = 64
OUTPUT_DIR        = f"../embeddings/{MODEL}"

# ── SETUP ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp", ".tiff", ".tif"}

if torch.cuda.is_available():
    DEVICE = "cuda"
elif torch.backends.mps.is_available():
    DEVICE = "mps"
else:
    DEVICE = "cpu"

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def collect_image_paths(input_dir):
    paths = [p for p in sorted(Path(input_dir).rglob("*")) if p.suffix.lower() in SUPPORTED_EXTENSIONS]
    log.info(f"Found {len(paths)} images in {input_dir}")
    return paths


def embed_images(paths, model, preprocess):
    embeddings, valid_paths, failed = [], [], []

    for i in tqdm(range(0, len(paths), BATCH_SIZE), desc="Embedding"):
        batch_paths = paths[i : i + BATCH_SIZE]
        tensors, batch_valid = [], []

        for p in batch_paths:
            try:
                img = Image.open(p).convert("RGB")
                tensors.append(preprocess(img))
                batch_valid.append(str(p))
            except Exception as e:
                log.warning(f"Skipping {p}: {e}")
                failed.append(str(p))

        if not tensors:
            continue

        batch = torch.stack(tensors).to(DEVICE)
        with torch.no_grad():
            feats = model.encode_image(batch)
            feats = feats / feats.norm(dim=-1, keepdim=True)  # L2 normalize

        embeddings.append(feats.cpu().float().numpy())
        valid_paths.extend(batch_valid)

    if failed:
        log.warning(f"{len(failed)} images failed to load and were skipped.")

    return np.vstack(embeddings), valid_paths


def save_outputs(label, embeddings, paths):
    out = Path(OUTPUT_DIR)
    out.mkdir(parents=True, exist_ok=True)

    np.save(out / f"{label}_embeddings.npy", embeddings)
    log.info(f"Saved embeddings → {out / f'{label}_embeddings.npy'}  shape={embeddings.shape}")

    with open(out / f"{label}_paths.json", "w") as f:
        json.dump(paths, f, indent=2)
    log.info(f"Saved paths      → {out / f'{label}_paths.json'}")

    with open(out / f"{label}_metadata.json", "w") as f:
        json.dump({
            "label": label,
            "model": MODEL,
            "embedding_dim": int(embeddings.shape[1]),
            "num_images": int(embeddings.shape[0]),
            "normalized": True,
            "created_at": datetime.utcnow().isoformat() + "Z",
        }, f, indent=2)
    log.info(f"Saved metadata   → {out / f'{label}_metadata.json'}")


# ── MAIN ──────────────────────────────────────────────────────────────────────

log.info(f"Using device: {DEVICE}")
log.info(f"Loading CLIP model '{MODEL}'...")
model, preprocess = clip.load(MODEL, device=DEVICE)
model.eval()

for label, input_dir in [("harmful", HARMFUL_INPUT_DIR), ("safe", SAFE_INPUT_DIR)]:
    paths = collect_image_paths(input_dir)
    if not paths:
        log.warning(f"No images found in {input_dir}, skipping.")
        continue
    embeddings, valid_paths = embed_images(paths, model, preprocess)
    save_outputs(label, embeddings, valid_paths)

log.info("All done.")