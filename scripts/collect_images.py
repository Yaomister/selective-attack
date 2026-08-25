"""
collect_images.py

Collects all images for each pair into a single folder structure
ready to zip and upload to Overleaf.

Output structure:
    overleaf_images/
        {pair_id}/
            harmful.jpg
            safe.jpg
            safe_to_harmful_perturbed.png
            harmful_to_safe_perturbed.png

Edit the CONFIG block below, then run:
    python collect_images.py
"""

import shutil
import logging
from pathlib import Path

# ── CONFIG ────────────────────────────────────────────────────────────────────

SORTED_DIR       = "../sorted"          # contains pair folders with harmful.jpg + safe.jpg
ATTACK_RESULTS   = "../../experiments/attack_results_3"  # contains per-pair attack output folders
OUTPUT_DIR       = "./overleaf_images_run3" # where collected images will go

# ── SETUP ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── MAIN ──────────────────────────────────────────────────────────────────────

sorted_dir     = Path(SORTED_DIR)
attack_dir     = Path(ATTACK_RESULTS)
output_dir     = Path(OUTPUT_DIR)
output_dir.mkdir(parents=True, exist_ok=True)

missing = []
collected = 0

for pair_dir in sorted(sorted_dir.iterdir()):
    if not pair_dir.is_dir():
        continue

    pair_id  = pair_dir.name
    pair_out = output_dir / pair_id
    pair_out.mkdir(exist_ok=True)

    # Original images from sorted/
    for fname in ["harmful.jpg", "safe.jpg"]:
        src = pair_dir / fname
        if src.exists():
            shutil.copy2(src, pair_out / fname)
        else:
            missing.append(f"{pair_id}/{fname}")

    # Perturbed images from attack_results/
    for direction, out_name in [
        ("safe_to_harmful",  "safe_to_harmful_perturbed.png"),
        ("harmful_to_safe",  "harmful_to_safe_perturbed.png"),
    ]:
        src = attack_dir / pair_id / direction / "perturbed.png"
        if src.exists():
            shutil.copy2(src, pair_out / out_name)
        else:
            missing.append(f"{pair_id}/{out_name}")

    collected += 1
    log.info(f"Collected pair {pair_id}")

# Zip the output folder
shutil.make_archive(str(output_dir), "zip", output_dir.parent, output_dir.name)
log.info(f"Zipped → {output_dir}.zip")

log.info(f"\nDone. {collected} pairs collected.")
if missing:
    log.warning(f"{len(missing)} missing files:")
    for m in missing:
        log.warning(f"  {m}")