"""
retrieve_matches.py

Loads precomputed CLIP embeddings, finds the top-k closest safe images
for each harmful image using FAISS, and outputs:
  - matches.json       grouped results with image names and scores
  - matches.csv        flat table for spreadsheet review
  - report.html        visual report with images displayed side by side
"""

import json
import csv
import base64
import logging
import numpy as np
import faiss
from pathlib import Path
from PIL import Image

# ── CONFIG ────────────────────────────────────────────────────────────────────

EMBEDDINGS_DIR   = "../embeddings/ViT-L/14"     # folder from generate_embeddings.py
HARMFUL_IMG_DIR  = "../harmful_images" # original harmful images folder
SAFE_IMG_DIR     = "../safe_images"    # original safe images folder
OUTPUT_DIR       = "../matches"        # where results will be saved

TOP_K            = 5                  # number of safe matches per harmful image
MIN_SIMILARITY   = 0.20              # drop matches below this cosine similarity

# ── SETUP ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

# ── FUNCTIONS ─────────────────────────────────────────────────────────────────

def load_embeddings(label):
    base = Path(EMBEDDINGS_DIR)
    embeddings = np.load(base / f"{label}_embeddings.npy").astype("float32")
    with open(base / f"{label}_paths.json") as f:
        paths = json.load(f)
    log.info(f"Loaded {len(paths)} {label} embeddings  shape={embeddings.shape}")
    return embeddings, paths


def build_index(embeddings):
    index = faiss.IndexFlatIP(embeddings.shape[1])
    index.add(embeddings)
    log.info(f"FAISS index built  ({index.ntotal} vectors)")
    return index


def retrieve(harmful_embeddings, harmful_paths, safe_paths, index):
    distances, indices = index.search(harmful_embeddings, TOP_K)
    results = []
    for i, (dists, idxs) in enumerate(zip(distances, indices)):
        for rank, (dist, idx) in enumerate(zip(dists, idxs), start=1):
            score = float(dist)
            if score < MIN_SIMILARITY:
                continue
            results.append({
                "harmful_idx":   i,
                "harmful_image": Path(harmful_paths[i]).name,
                "harmful_path":  harmful_paths[i],
                "rank":          rank,
                "safe_image":    Path(safe_paths[idx]).name,
                "safe_path":     safe_paths[idx],
                "similarity":    round(score, 4),
                "selected":      "",
            })
    log.info(f"Retrieved {len(results)} matches for {len(harmful_paths)} harmful images")
    return results


def save_json(results, path):
    grouped = {}
    for r in results:
        idx = r["harmful_idx"]
        if idx not in grouped:
            grouped[idx] = {"harmful_idx": idx, "harmful_image": r["harmful_image"], "matches": []}
        grouped[idx]["matches"].append({
            "rank": r["rank"], "safe_image": r["safe_image"],
            "similarity": r["similarity"], "selected": r["selected"],
        })
    with open(path, "w") as f:
        json.dump(list(grouped.values()), f, indent=2)
    log.info(f"Saved JSON → {path}")


def save_csv(results, path):
    fieldnames = ["harmful_idx", "harmful_image", "rank", "safe_image", "similarity", "selected"]
    with open(path, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for r in results:
            writer.writerow({k: r[k] for k in fieldnames})
    log.info(f"Saved CSV  → {path}")


def img_to_b64(img_path, max_size=160):
    """Read an image, resize to thumbnail, and return a base64 data URI."""
    import io
    p = Path(img_path)
    if not p.exists():
        for search_dir in [HARMFUL_IMG_DIR, SAFE_IMG_DIR]:
            candidate = Path(search_dir) / p.name
            if candidate.exists():
                p = candidate
                break
    if not p.exists():
        return None
    img = Image.open(p).convert("RGB")
    img.thumbnail((max_size, max_size), Image.LANCZOS)
    buf = io.BytesIO()
    img.save(buf, format="JPEG", quality=70)
    data = base64.b64encode(buf.getvalue()).decode("utf-8")
    return f"data:image/jpeg;base64,{data}"


def save_html(results, path):
    # Group by harmful image
    grouped = {}
    for r in results:
        idx = r["harmful_idx"]
        if idx not in grouped:
            grouped[idx] = r
            grouped[idx]["matches"] = []
        grouped[idx]["matches"].append(r)

    # Build a flat JS lookup: card_id -> {harmful_idx, harmful_path, safe_path}
    pairs_lookup = {}
    rows_html = ""
    for idx, entry in sorted(grouped.items()):
        harmful_uri = img_to_b64(entry["harmful_path"])
        harmful_img = f'<img src="{harmful_uri}" alt="{entry["harmful_image"]}">' if harmful_uri else "<div class='no-img'>Image not found</div>"

        matches_html = ""
        for m in entry["matches"]:
            card_id  = f"{idx}_{m['rank']}"
            safe_uri = img_to_b64(m["safe_path"])
            safe_img = f'<img src="{safe_uri}" alt="{m["safe_image"]}">' if safe_uri else "<div class='no-img'>Image not found</div>"
            score_class = "score-high" if m["similarity"] >= 0.35 else "score-mid" if m["similarity"] >= 0.25 else "score-low"

            pairs_lookup[card_id] = {
                "harmful_idx":  idx,
                "harmful_path": str(Path(entry["harmful_path"]).resolve()),
                "safe_path":    str(Path(m["safe_path"]).resolve()),
            }

            matches_html += f"""
            <div class="match-card" data-card-id="{card_id}">
                {safe_img}
                <div class="match-meta">
                    <span class="rank">#{m['rank']}</span>
                    <span class="score {score_class}">{m['similarity']:.3f}</span>
                </div>
                <div class="match-name">{m['safe_image']}</div>
                <div class="selected-badge">✓ Selected</div>
            </div>"""

        rows_html += f"""
        <div class="row" id="row-{idx}">
            <div class="harmful-col">
                <div class="label">HARMFUL</div>
                {harmful_img}
                <div class="img-name">{entry['harmful_image']}</div>
                <div class="idx">#{idx}</div>
            </div>
            <div class="arrow">→</div>
            <div class="matches-col">
                {matches_html}
            </div>
            <div class="row-status" id="status-{idx}"></div>
        </div>"""

    pairs_json = json.dumps(pairs_lookup)

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Match Report</title>
<style>
  * {{ box-sizing: border-box; margin: 0; padding: 0; }}
  body {{ font-family: system-ui, sans-serif; background: #0f0f0f; color: #e0e0e0; padding: 24px; }}
  h1 {{ font-size: 1.3rem; font-weight: 600; margin-bottom: 6px; color: #fff; }}
  .subtitle {{ font-size: 0.85rem; color: #666; margin-bottom: 8px; }}
  .progress-bar-wrap {{ background: #1e1e1e; border-radius: 6px; height: 6px; width: 300px; margin-bottom: 28px; }}
  .progress-bar {{ background: #2ecc71; height: 6px; border-radius: 6px; transition: width 0.3s; }}

  .row {{
    display: flex; align-items: center; gap: 16px;
    background: #1a1a1a; border: 1px solid #2a2a2a;
    border-radius: 10px; padding: 16px; margin-bottom: 16px;
    transition: border-color 0.2s;
  }}
  .row:hover {{ border-color: #444; }}
  .row.done {{ border-color: #1a4a2a; background: #141f17; }}

  .harmful-col {{
    display: flex; flex-direction: column; align-items: center;
    gap: 6px; min-width: 140px; max-width: 140px;
  }}
  .harmful-col img {{ width: 130px; height: 130px; object-fit: cover; border-radius: 6px; border: 2px solid #c0392b; }}
  .label {{ font-size: 0.65rem; font-weight: 700; letter-spacing: 0.08em;
            color: #c0392b; background: #2a1010; padding: 2px 8px; border-radius: 4px; }}
  .img-name {{ font-size: 0.7rem; color: #888; text-align: center; word-break: break-all; }}
  .idx {{ font-size: 0.65rem; color: #555; }}

  .arrow {{ font-size: 1.4rem; color: #444; flex-shrink: 0; }}
  .matches-col {{ display: flex; gap: 10px; flex-wrap: wrap; flex: 1; }}

  .match-card {{
    display: flex; flex-direction: column; align-items: center;
    gap: 5px; background: #222; border: 2px solid #2e2e2e;
    border-radius: 8px; padding: 8px; min-width: 110px; max-width: 110px;
    cursor: pointer; transition: border-color 0.15s, background 0.15s;
    position: relative;
  }}
  .match-card:hover {{ border-color: #555; background: #2a2a2a; }}
  .match-card.selected {{ border-color: #2ecc71; background: #0d1f13; }}
  .match-card img {{ width: 100px; height: 100px; object-fit: cover; border-radius: 5px; }}
  .match-meta {{ display: flex; gap: 6px; align-items: center; }}
  .rank {{ font-size: 0.7rem; color: #666; }}
  .score {{ font-size: 0.75rem; font-weight: 600; padding: 1px 6px; border-radius: 4px; }}
  .score-high {{ background: #0d3320; color: #2ecc71; }}
  .score-mid  {{ background: #2d2500; color: #f1c40f; }}
  .score-low  {{ background: #2a1a00; color: #e67e22; }}
  .match-name {{ font-size: 0.65rem; color: #777; text-align: center; word-break: break-all; }}
  .no-img {{ width: 100px; height: 100px; background: #2a2a2a; border-radius: 5px;
             display: flex; align-items: center; justify-content: center;
             font-size: 0.65rem; color: #555; text-align: center; padding: 4px; }}

  .selected-badge {{
    display: none; font-size: 0.65rem; font-weight: 700; color: #2ecc71;
    background: #0d3320; padding: 2px 8px; border-radius: 4px; letter-spacing: 0.05em;
  }}
  .match-card.selected .selected-badge {{ display: block; }}

  .row-status {{
    font-size: 0.7rem; color: #2ecc71; min-width: 60px; text-align: center;
  }}

  #toast {{
    position: fixed; bottom: 24px; right: 24px;
    background: #1e3a28; color: #2ecc71; border: 1px solid #2ecc71;
    padding: 10px 18px; border-radius: 8px; font-size: 0.8rem;
    opacity: 0; transition: opacity 0.3s; pointer-events: none;
    z-index: 1000;
  }}
  #toast.show {{ opacity: 1; }}
</style>
</head>
<body>
  <h1>Match Report</h1>
  <div class="subtitle">
    <span id="selected-count">0</span> / {len(grouped)} selected
    &nbsp;·&nbsp; top {TOP_K} safe matches each
    &nbsp;·&nbsp; score: <span style="color:#2ecc71">≥0.35 high</span>
    &nbsp;·&nbsp; <span style="color:#f1c40f">≥0.25 mid</span>
    &nbsp;·&nbsp; <span style="color:#e67e22">below low</span>
  </div>
  <div class="progress-bar-wrap">
    <div class="progress-bar" id="progress-bar" style="width:0%"></div>
  </div>

  {rows_html}

  <div id="toast"></div>

<script>
  const TOTAL = {len(grouped)};
  let selectedCount = 0;

  // Store all paths in a JS lookup — avoids any issues with special chars in data attributes
  const PAIR_DATA = {pairs_json};

  document.querySelectorAll('.match-card').forEach(card => {{
    card.addEventListener('click', function() {{
      const cardId      = this.dataset.cardId;
      const pair        = PAIR_DATA[cardId];
      const harmfulIdx  = pair.harmful_idx;
      const row         = document.getElementById('row-' + harmfulIdx);
      const allCards    = row.querySelectorAll('.match-card');
      const wasSelected = this.classList.contains('selected');

      if (wasSelected) {{
        this.classList.remove('selected');
        row.classList.remove('done');
        document.getElementById('status-' + harmfulIdx).textContent = '';
        fetch('/deselect', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ harmful_idx: harmfulIdx }})
        }});
        selectedCount--;
        updateProgress();
        showToast('Deselected pair #' + harmfulIdx);
      }} else {{
        const wasAlreadyDone = row.classList.contains('done');
        allCards.forEach(c => c.classList.remove('selected'));
        this.classList.add('selected');
        row.classList.add('done');
        document.getElementById('status-' + harmfulIdx).textContent = '✓';
        fetch('/select', {{
          method: 'POST',
          headers: {{'Content-Type': 'application/json'}},
          body: JSON.stringify({{ harmful_idx: harmfulIdx, harmful_path: pair.harmful_path, safe_path: pair.safe_path }})
        }})
        .then(r => r.json())
        .then(() => showToast('Saved pair #' + harmfulIdx))
        .catch(err => showToast('Error: ' + err));
        if (!wasAlreadyDone) selectedCount++;
        updateProgress();
      }}
    }});
  }});

  // Restore already-selected pairs on load
  fetch('/status')
    .then(r => r.json())
    .then(data => {{
      data.selected.forEach(idx => {{
        const row = document.getElementById('row-' + idx);
        if (!row) return;
        row.classList.add('done');
        document.getElementById('status-' + idx).textContent = '✓';
        selectedCount++;
      }});
      updateProgress();
    }});

  function updateProgress() {{
    document.getElementById('selected-count').textContent = selectedCount;
    document.getElementById('progress-bar').style.width = (selectedCount / TOTAL * 100) + '%';
  }}

  function showToast(msg) {{
    const t = document.getElementById('toast');
    t.textContent = msg;
    t.classList.add('show');
    setTimeout(() => t.classList.remove('show'), 2000);
  }}
</script>
</body>
</html>"""

    with open(path, "w") as f:
        f.write(html)
    log.info(f"Saved HTML → {path}")


def print_summary(results, harmful_paths):
    total    = len(harmful_paths)
    matched  = len({r["harmful_idx"] for r in results})
    avg_top1 = np.mean([r["similarity"] for r in results if r["rank"] == 1])
    print("\n── Match Summary ─────────────────────────────────────────")
    print(f"  Harmful images total  : {total}")
    print(f"  Images with matches   : {matched}")
    print(f"  Images with no match  : {total - matched}")
    print(f"  Avg top-1 similarity  : {avg_top1:.4f}")
    print(f"  Total pairs saved     : {len(results)}")
    print("──────────────────────────────────────────────────────────\n")


# ── MAIN ──────────────────────────────────────────────────────────────────────

harmful_embeddings, harmful_paths = load_embeddings("harmful")
safe_embeddings,    safe_paths    = load_embeddings("safe")

index   = build_index(safe_embeddings)
results = retrieve(harmful_embeddings, harmful_paths, safe_paths, index)

Path(OUTPUT_DIR).mkdir(parents=True, exist_ok=True)
save_json(results, Path(OUTPUT_DIR) / "matches.json")
save_csv(results,  Path(OUTPUT_DIR) / "matches.csv")
save_html(results, Path(OUTPUT_DIR) / "report.html")

print_summary(results, harmful_paths)
log.info("Done.")