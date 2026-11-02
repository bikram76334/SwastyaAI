"""
inference.py

Loads the trained DR models lazily and exposes predict_dr(image_bytes, model_key)
for run.py to call.

Two model kinds are supported:

  kind "coral"      -> CoralEffNet (EfficientNet backbone + CORAL ordinal head,
                       trained in bo-dr-2024.ipynb). Uses transforms.preprocess_bytes.
  kind "multitask"  -> MultiTaskEffNet (EfficientNet-B3, 3 heads: laterality,
                       centre, DR grade; trained in the effb3-aptos-multimodal
                       notebook). Uses its OWN preprocessing (black-border crop,
                       square pad, 512px, Ben Graham + CLAHE, ImageNet normalize),
                       copied from that notebook. Only the DR head is used here.

CORAL recap: the DR head does NOT output 5 class scores like a softmax
classifier. It outputs 4 threshold logits - P(grade > 0), P(grade > 1),
P(grade > 2), P(grade > 3). The predicted grade is how many thresholds are
crossed (sigmoid > 0.5), and class probabilities are the differences between
consecutive thresholds.

To add another family/seed: add an entry in MODEL_FAMILIES. Keep the returned
dict shape the same so dr.html does not need to change.
"""

import os

import cv2
import numpy as np
import torch
import torch.nn as nn
import torchvision

from transforms import preprocess_bytes

# ---------------------------------------------------------------- paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DR_DIR = os.path.join(BASE_DIR, "models", "dr")

# One entry per model family. "kind" picks architecture + preprocessing,
# "arch" is the torchvision constructor name. Dropdown values look like
# "e1b0:2024" (single seed) or "e1b0:all" (ensemble of every seed).
MODEL_FAMILIES = {
    "e1b0": {
        "label": "E1 with B0",
        "kind": "coral",
        "arch": "efficientnet_b0",
        "seeds": {
            "2024": os.path.join(DR_DIR, "e1withB0", "seed_2024", "e1_b0_seed2024.pt"),
            "42":   os.path.join(DR_DIR, "e1withB0", "seed_42",   "e1_b0_seed42.pt"),
            "123":  os.path.join(DR_DIR, "e1withB0", "seed_123",  "e1_b0_seed123.pt"),
        },
    },
  
    "effb3": {
        "label": "EfficientNet-B3 (multi-task)",
        "kind": "multitask",
        "arch": "efficientnet_b3",
        "seeds": {
            "42": os.path.join(DR_DIR, "efficentnetb3", "effb3_multitask_bundle.pt"),
        },
    },
}

# Grouped dropdown: {"E1 with B0": {"e1b0:2024": "Seed 2024", ...}, ...}
MODEL_CHOICES = {}
_LABELS = {}  # value -> full label shown in the result card
for _fam_key, _fam in MODEL_FAMILIES.items():
    _group = {f"{_fam_key}:{seed}": f"Seed {seed}" for seed in _fam["seeds"]}

    MODEL_CHOICES[_fam["label"]] = _group
    for _value, _label in _group.items():
        _LABELS[_value] = _fam["label"] + " - " + _label
    if len(_fam["seeds"]) > 1:
        _LABELS[f"{_fam_key}:all"] = _fam["label"] + " - Ensemble"

# Must be a value that exists in the dropdown 
DEFAULT_MODEL = "e1b0:2024"

METRIC_CHOICES = [
    ("quantitative", ""),
    ("qualittaive", ""),
]

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A result is flagged low-confidence if the model's probability on its
# own predicted class is below this.
UNCERTAIN_THRESHOLD = 0.5


# ---------------------------------------------------------------- models
class CoralEffNet(nn.Module):
    """Same architecture as bo-dr-2024.ipynb (cell 6). weights=None - we load
    our own trained state_dict, so no internet is needed at startup."""

    def __init__(self, arch="efficientnet_b0"):
        super().__init__()
        base = getattr(torchvision.models, arch)(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.fc = nn.Linear(1280, 1, bias=False)
        self.bias = nn.Parameter(torch.linspace(1.5, -1.5, 4))

    def forward(self, x):
        h = self.pool(self.features(x)).flatten(1)
        z = self.fc(self.drop(h))
        return z + self.bias


class MultiTaskEffNet(nn.Module):
    """Same architecture as the B3 notebook (cell 17). forward returns
    (laterality logits, centre logits, DR CORAL logits). B3 ends in 1536 features."""

    def __init__(self, arch="efficientnet_b3"):
        super().__init__()
        base = getattr(torchvision.models, arch)(weights=None)
        self.features = base.features
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.drop = nn.Dropout(0.3)
        self.lat_head = nn.Linear(1536, 2)
        self.cen_head = nn.Linear(1536, 2)
        self.dr_fc = nn.Linear(1536, 1, bias=False)
        self.dr_bias = nn.Parameter(torch.linspace(1.5, -1.5, 4))

    def forward(self, x):
        h = self.drop(self.pool(self.features(x)).flatten(1))
        return self.lat_head(h), self.cen_head(h), self.dr_fc(h) + self.dr_bias


# ---------------------------------------------------------------- B3 preprocessing
# Copied from the B3 notebook (cell 9) so inference matches training exactly.
_MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
_STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)


def _crop_black(img):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    ys, xs = np.where(gray > 10)
    if len(xs) == 0:
        return img
    return img[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _make_square(img):
    h, w = img.shape[:2]
    d = abs(h - w)
    a, b = d // 2, d - d // 2
    if h > w:
        return cv2.copyMakeBorder(img, 0, 0, a, b, cv2.BORDER_CONSTANT, value=0)
    return cv2.copyMakeBorder(img, a, b, 0, 0, cv2.BORDER_CONSTANT, value=0)


def _enhance(img, size, prep):
    if prep in ("ben", "both"):
        blur = cv2.GaussianBlur(img, (0, 0), size / 2 / 30)
        img = cv2.addWeighted(img, 4, blur, -4, 128)
    if prep in ("clahe", "both"):
        lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
        lab[:, :, 0] = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8)).apply(lab[:, :, 0])
        img = cv2.cvtColor(lab, cv2.COLOR_LAB2RGB)
    return img


def preprocess_b3_bytes(image_bytes, size=512, prep="both"):
    """Uploaded bytes -> normalized tensor of shape (1, 3, size, size)."""
    arr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Could not decode image")
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(_make_square(_crop_black(img)), (size, size))
    img = _enhance(img, size, prep)
    x = (img.astype(np.float32) / 255.0 - _MEAN) / _STD
    return torch.from_numpy(x).permute(2, 0, 1).unsqueeze(0)


# ---------------------------------------------------------------- loading
_models = {}  # (family key, seed) -> (model, meta)


def _get_model(fam_key, seed):
    key = (fam_key, seed)
    if key not in _models:
        fam = MODEL_FAMILIES[fam_key]
        obj = torch.load(fam["seeds"][seed], map_location=DEVICE)

        # the B3 notebook saves either a plain state_dict or a bundle dict
        meta = {"img": 512, "prep": "both", "temp": 1.0}
        if isinstance(obj, dict) and "state_dict" in obj:
            meta["img"] = int(obj.get("IMG", 512))
            meta["prep"] = obj.get("PREP", "both")
            meta["temp"] = float(obj.get("TEMPS", {}).get("dr", 1.0))
            state_dict = obj["state_dict"]
        else:
            state_dict = obj

        if fam["kind"] == "multitask":
            model = MultiTaskEffNet(fam["arch"]).to(DEVICE)
        else:
            model = CoralEffNet(fam["arch"]).to(DEVICE)
        model.load_state_dict(state_dict)
        model.eval()
        _models[key] = (model, meta)
    return _models[key]


def _thresholds_to_class_probs(p):
    """p: (4,) sigmoid outputs of the CORAL thresholds -> (5,) class probabilities."""
    cp = np.zeros(5)
    cp[0] = 1 - p[0]
    for k in range(1, 4):
        cp[k] = p[k - 1] - p[k]
    cp[4] = p[3]
    cp = np.clip(cp, 1e-6, None)
    cp = cp / cp.sum()
    return cp


def predict_dr(image_bytes, model_key=DEFAULT_MODEL):
    """Runs the full pipeline on one uploaded image.
    model_key: "<family>:<seed>" for one seed, or "<family>:all" to average the
    CORAL threshold probabilities of every seed in that family.
    Returns: predicted_class, predicted_grade, confidence,
             classes: [{name, probability}, ...], is_uncertain, model_used
    """
    try:
        fam_key, which = model_key.split(":")
        fam = MODEL_FAMILIES[fam_key]
    except (ValueError, KeyError, AttributeError):
        raise ValueError("Unknown model: " + str(model_key))

    if which == "all":
        seeds = list(fam["seeds"].keys())
    elif which in fam["seeds"]:
        seeds = [which]
    else:
        raise ValueError("Unknown model: " + str(model_key))

    per_seed_p = []
    x_coral = None  # preprocess once for the original models
    with torch.no_grad():
        for s in seeds:
            model, meta = _get_model(fam_key, s)
            if fam["kind"] == "multitask":
                x = preprocess_b3_bytes(image_bytes, meta["img"], meta["prep"]).to(DEVICE)
                dr_logits = model(x)[2]  # (laterality, centre, DR) -> DR only
                p = torch.sigmoid(dr_logits / meta["temp"]).cpu().numpy()[0]
            else:
                if x_coral is None:
                    x_coral = preprocess_bytes(image_bytes).to(DEVICE)
                p = torch.sigmoid(model(x_coral)).cpu().numpy()[0]
            per_seed_p.append(p)

    p_mean = np.mean(per_seed_p, axis=0)  # with one model this is just that model

    predicted_grade = int((p_mean > 0.5).sum())
    class_probs = _thresholds_to_class_probs(p_mean)
    confidence = float(class_probs[predicted_grade])

    return {
        "predicted_class": CLASS_NAMES[predicted_grade],
        "predicted_grade": predicted_grade,
        "confidence": confidence,
        "classes": [
            {"name": name, "probability": float(prob)}
            for name, prob in zip(CLASS_NAMES, class_probs)
        ],
        "is_uncertain": confidence < UNCERTAIN_THRESHOLD,
        "model_used": _LABELS[model_key],
    }