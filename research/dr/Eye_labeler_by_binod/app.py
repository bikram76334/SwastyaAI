#!/usr/bin/env python3
"""
Fundus Eye-Side Labeler  (Left / Right / Unsure)
------------------------------------------------
Usage:
    python app.py                       # downloads dataset via kagglehub (first run only), then serves
    python app.py --root "D:/data/dr"   # use an already-downloaded copy
    python app.py --include aptos multimodal --exclude mask segmentation

Open http://127.0.0.1:5000
Everything you label is saved instantly to  output/labels.csv  (atomic write).
"""
import argparse, csv, os, shutil, threading
from datetime import datetime
from pathlib import Path
from flask import Flask, Response, abort, jsonify, render_template, request, send_file

KAGGLE_SLUG = "deepapaneru/diabetic-retinopathy-datasets"
IMG_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
VALID = {"left", "right", "unsure"}
FIELDS = ["image_path", "dataset", "label", "labeled_at"]

BASE = Path(__file__).parent.resolve()
OUT = BASE / "output"
CSV_PATH = OUT / "labels.csv"

app = Flask(__name__)
lock = threading.Lock()
ROOT: Path = None
INDEX = []        # list of (rel_path, dataset) – stable sorted order
PATH2ID = {}
LABELS = {}       # rel_path -> row dict


def resolve_root(arg):
    if arg:
        return Path(arg).expanduser().resolve()
    import kagglehub  # downloads once, then reuses local cache
    print("Fetching dataset from Kaggle (first run only, can be large)...")
    return Path(kagglehub.dataset_download(KAGGLE_SLUG)).resolve()


def scan(root, include, exclude):
    found, tops = [], {}
    for dp, _, files in os.walk(root):
        for f in files:
            if Path(f).suffix.lower() not in IMG_EXT:
                continue
            rel = (Path(dp) / f).relative_to(root).as_posix()
            low = rel.lower()
            top = rel.split("/")[0]
            tops[top] = tops.get(top, 0) + 1
            if any(x.lower() in low for x in exclude):
                continue
            ds = next((k for k in include if k.lower() in low), None)
            if ds:
                found.append((rel, ds.upper()))
    found.sort(key=lambda t: t[0])
    return found, tops


def load_csv():
    if not CSV_PATH.exists():
        return
    with open(CSV_PATH, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            if r.get("image_path") and r.get("label") in VALID:
                LABELS[r["image_path"]] = {k: r.get(k, "") for k in FIELDS}


def save_csv():
    """Atomic write: temp file -> fsync -> rename. A crash can never corrupt labels.csv."""
    tmp = CSV_PATH.with_suffix(".tmp")
    with open(tmp, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, FIELDS)
        w.writeheader()
        for k in sorted(LABELS):
            w.writerow(LABELS[k])
        f.flush()
        os.fsync(f.fileno())
    os.replace(tmp, CSV_PATH)


def stats():
    s = {"total": len(INDEX), "left": 0, "right": 0, "unsure": 0, "by_dataset": {}}
    for rel, ds in INDEX:
        d = s["by_dataset"].setdefault(ds, {"total": 0, "done": 0})
        d["total"] += 1
        row = LABELS.get(rel)
        if row:
            s[row["label"]] += 1
            d["done"] += 1
    s["done"] = s["left"] + s["right"] + s["unsure"]
    return s


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/api/items")
def items():
    ds, st = request.args.get("dataset", "all"), request.args.get("status", "all")
    out = []
    for i, (rel, d) in enumerate(INDEX):
        if ds != "all" and d != ds:
            continue
        lab = LABELS.get(rel, {}).get("label")
        if st == "unlabeled" and lab: continue
        if st == "labeled" and not lab: continue
        if st in VALID and lab != st: continue
        out.append({"id": i, "path": rel, "ds": d, "label": lab})
    return jsonify(items=out, stats=stats(), datasets=sorted({d for _, d in INDEX}))


@app.route("/api/label", methods=["POST"])
def label():
    j = request.get_json(force=True)
    rel, lab = j.get("path"), j.get("label")
    if rel not in PATH2ID or (lab is not None and lab not in VALID):
        abort(400)
    with lock:
        if lab is None:
            LABELS.pop(rel, None)
        else:
            LABELS[rel] = {"image_path": rel, "dataset": INDEX[PATH2ID[rel]][1],
                           "label": lab, "labeled_at": datetime.now().isoformat(timespec="seconds")}
        save_csv()
        return jsonify(ok=True, stats=stats())


@app.route("/img/<int:i>")
def img(i):
    if not 0 <= i < len(INDEX):
        abort(404)
    return send_file(ROOT / INDEX[i][0], max_age=3600)


@app.route("/export/train.csv")
def export_train():
    """Clean file for training: image_path,label,label_id (left=0, right=1). 'unsure' excluded."""
    lines = ["image_path,label,label_id"]
    for k in sorted(LABELS):
        r = LABELS[k]
        if r["label"] in ("left", "right"):
            p = '"' + k.replace('"', '""') + '"'
            lines.append(f'{p},{r["label"]},{0 if r["label"] == "left" else 1}')
    return Response("\n".join(lines) + "\n", mimetype="text/csv",
                    headers={"Content-Disposition": "attachment; filename=labels_train.csv"})


@app.route("/export/labels.csv")
def export_all():
    return send_file(CSV_PATH, as_attachment=True) if CSV_PATH.exists() else abort(404)


def main():
    global ROOT, INDEX, PATH2ID
    ap = argparse.ArgumentParser()
    ap.add_argument("--root")
    ap.add_argument("--include", nargs="+", default=["aptos", "multimodal"])
    ap.add_argument("--exclude", nargs="*", default=[])
    ap.add_argument("--port", type=int, default=5000)
    a = ap.parse_args()

    ROOT = resolve_root(a.root)
    INDEX, tops = scan(ROOT, a.include, a.exclude)
    PATH2ID = {r: i for i, (r, _) in enumerate(INDEX)}
    print(f"\nDataset root: {ROOT}\nTop-level folders (image counts):")
    for k, v in sorted(tops.items()):
        print(f"  {k}: {v}")
    print(f"\nMatched {len(INDEX)} images for {a.include}  (exclude={a.exclude})")
    if not INDEX:
        raise SystemExit("No images matched. Check folder names above and pass --include accordingly.")

    OUT.mkdir(exist_ok=True)
    load_csv()
    if CSV_PATH.exists():  # daily safety backup
        (OUT / "backups").mkdir(exist_ok=True)
        shutil.copy(CSV_PATH, OUT / "backups" / f"labels_{datetime.now():%Y%m%d_%H%M%S}.csv")
    print(f"Resuming with {len(LABELS)} labels already saved.\nOpen http://127.0.0.1:{a.port}\n")
    app.run(host="127.0.0.1", port=a.port, threaded=True)


if __name__ == "__main__":
    main()
