# Fundus Eye-Side Labeler

## Setup (once)
1. `pip install -r requirements.txt`
2. Kaggle credentials: kaggle.com > Settings > API > Create New Token, save `kaggle.json` to
   `~/.kaggle/kaggle.json` (Windows: `C:\Users\<you>\.kaggle\kaggle.json`).

## Run
    python app.py
First run downloads/caches the Kaggle dataset automatically and prints its folder structure.
If APTOS / Multimodal folders are not detected, use `--include` / `--exclude` (see header of app.py).
Already downloaded? `python app.py --root "D:\path\to\dataset"`.

## Daily use
Open http://127.0.0.1:5000 - it opens at the first unlabeled image. Close anytime; nothing is lost.

Keys: L = Left, R = Right, U = Unsure, <- / -> = navigate, Backspace = clear label, Z = zoom.

## Output (`output/labels.csv`)
    image_path,dataset,label,labeled_at
Paths are relative to the dataset root, forward slashes (works on Windows/Linux).
`/export/train.csv` gives `image_path,label,label_id` (left=0, right=1, unsure removed).

    df = pd.read_csv("labels_train.csv"); df["full"] = ROOT + "/" + df.image_path
