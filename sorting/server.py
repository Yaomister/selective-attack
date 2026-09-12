"""
server.py

Locally hosted Flask server for reviewing and selecting image pairs.
Serves the report and handles copying selected pairs to the output folder.


Then open http://127.0.0.1:5000 in your browser.
"""

import shutil
import logging
from pathlib import Path
from flask import Flask, request, jsonify, send_from_directory, abort

# ── CONFIG ────────────────────────────────────────────────────────────────────

MATCHES_DIR      = "../matches"        # folder containing report.html + matches.json
HARMFUL_IMG_DIR  = "../harmful_images" # original harmful images folder
SAFE_IMG_DIR     = "../safe_images"    # original safe images folder
SORTED_DIR       = "../sorted"         # where selected pairs will be copied to
PORT             = 5000

# ── SETUP ─────────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", datefmt="%H:%M:%S")
log = logging.getLogger(__name__)

app = Flask(__name__)

# ── ROUTES ────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return send_from_directory(Path(MATCHES_DIR).resolve(), "report.html")


@app.route("/images/harmful/<path:filename>")
def serve_harmful(filename):
    return send_from_directory(Path(HARMFUL_IMG_DIR).resolve(), filename)


@app.route("/images/safe/<path:filename>")
def serve_safe(filename):
    return send_from_directory(Path(SAFE_IMG_DIR).resolve(), filename)


@app.route("/select", methods=["POST"])
def select():
    data         = request.json
    harmful_idx  = data["harmful_idx"]
    harmful_path = Path(data["harmful_path"])
    safe_path    = Path(data["safe_path"])

    if not harmful_path.exists():
        abort(404, f"Harmful image not found: {harmful_path}")
    if not safe_path.exists():
        abort(404, f"Safe image not found: {safe_path}")

    pair_dir = Path(SORTED_DIR) / str(harmful_idx)
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
    pair_dir.mkdir(parents=True)

    shutil.copy2(harmful_path, pair_dir / f"harmful{harmful_path.suffix}")
    shutil.copy2(safe_path,    pair_dir / f"safe{safe_path.suffix}")

    log.info(f"Pair {harmful_idx} selected → {pair_dir}")
    return jsonify({"status": "ok", "pair_dir": str(pair_dir)})


@app.route("/deselect", methods=["POST"])
def deselect():
    harmful_idx = request.json["harmful_idx"]
    pair_dir = Path(SORTED_DIR) / str(harmful_idx)
    if pair_dir.exists():
        shutil.rmtree(pair_dir)
        log.info(f"Pair {harmful_idx} deselected, folder removed.")
    return jsonify({"status": "ok"})


@app.route("/status", methods=["GET"])
def status():
    sorted_path = Path(SORTED_DIR)
    if not sorted_path.exists():
        return jsonify({"selected": []})
    selected = [int(p.name) for p in sorted_path.iterdir() if p.is_dir()]
    return jsonify({"selected": selected})


# ── MAIN ──────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    Path(SORTED_DIR).mkdir(parents=True, exist_ok=True)
    log.info(f"Sorted pairs will be saved to: {Path(SORTED_DIR).resolve()}")
    log.info(f"Open http://127.0.0.1:{PORT} in your browser")
    app.run(host="127.0.0.1", port=PORT, debug=True)