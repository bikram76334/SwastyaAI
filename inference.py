"""
inference.py

Loads the trained DR models (CoralEffNet: EfficientNet backbone + a CORAL
ordinal head, trained in bo-dr-2024.ipynb) lazily, and exposes
predict_dr(image_bytes, model_key) for run.py to call.



CORAL recap: this model does NOT output 5 class scores directly like a
normal softmax classifier. It outputs 4 threshold logits —
P(grade > 0), P(grade > 1), P(grade > 2), P(grade > 3) — sharing one
feature score with 4 learned biases (notebook cell 6). The predicted
grade is how many of those thresholds are crossed (sigmoid > 0.5), and
class probabilities are the differences between consecutive thresholds
(notebook cell 8). That conversion is reproduced exactly below so this
app's numbers match the notebook's own test results.



Same pattern for your other modules: duplicate this file (e.g.
inference_pneumonia.py) with that model's architecture and checkpoint
path. Keep the function name predict_<disease> and the same returned
dict shape, so dr.html's layout can be copied for the next page.
"""

import os

import numpy as np
import torch
import torch.nn as nn
import torchvision

from transforms import preprocess_bytes

#paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DR_DIR = os.path.join(BASE_DIR, "models", "dr")

# One entry per model family ..
# values ("e1b0:2024"). "arch" is the torchvision constructor name.
MODEL_FAMILIES = {
    "e1b0": {
        "label": "E1 with B0",
        "arch": "efficientnet_b0",
        "seeds": {
            "2024": os.path.join(DR_DIR, "e1withB0", "seed_2024", "e1_b0_seed2024.pt"),
            "42":   os.path.join(DR_DIR, "e1withB0", "seed_42",   "e1_b0_seed42.pt"),
            "123":  os.path.join(DR_DIR, "e1withB0", "seed_123",  "e1_b0_seed123.pt"),
        },
    },
    # Add B1 later 
     "e1b1": {
         "label": "E1 with B1",
         "arch": "efficientnet_b1",
        "seeds": {
             "2024": os.path.join(DR_DIR, "e1withB1", "seed_2024", "e1_b1_seed2024.pt"),
            "42":   os.path.join(DR_DIR, "e1withB1", "seed_42",   "e1_b1_seed42.pt"),
            "123":  os.path.join(DR_DIR, "e1withB1", "seed_123",  "e1_b1_seed123.pt"),
        },
    },
}

# Built from MODEL_FAMILIES. Dictionary shape, used for the grouped dropdown:
#   {
#     "E1 with B0": {
#         "e1b0:2024": "Seed 2024",
#         "e1b0:42":   "Seed 42",
#         "e1b0:123":  "Seed 123",
#         "e1b0:all":  "Ensemble (all 3 seeds)",
#     },
#   }
MODEL_CHOICES = {}
_LABELS = {}  # value -> full label shown in the result card
for _fam_key, _fam in MODEL_FAMILIES.items():
    _group = {f"{_fam_key}:{seed}": f"Seed {seed}" for seed in _fam["seeds"]}
    _group[f"{_fam_key}:all"] = f"Ensemble (all {len(_fam['seeds'])} seeds)"
    MODEL_CHOICES[_fam["label"]] = _group
    for _value, _label in _group.items():
        _LABELS[_value] = _fam["label"] + " - " + _label

DEFAULT_MODEL = "e1b0:all"

METRIC_CHOICES = [
    ("quantitative", ""),
    ("qualittaive", ""),

]

CLASS_NAMES = ["No DR", "Mild", "Moderate", "Severe", "Proliferative DR"]
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# A result is flagged low-confidence if the model's probability on its
# own predicted class is below this. 0.5 is a starting point — tune it
# once looked at real outputs on held-out images.
UNCERTAIN_THRESHOLD = 0.5


class CoralEffNet(nn.Module):
    """Same architecture as the notebook (cell 6). weights=None here —
    we load our own trained state_dict immediately below, so the app
    doesn't need internet access to download ImageNet weights on every
    startup (the notebook only needed those for the initial training
    run, not for inference). `arch` picks the EfficientNet variant;
    B0 and B1 both end in 1280 features, so the head is the same."""

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


_models = {}  # (family key, seed) 


def _get_model(fam_key, seed):
    key = (fam_key, seed)
    if key not in _models:
        fam = MODEL_FAMILIES[fam_key]
        model = CoralEffNet(fam["arch"]).to(DEVICE)
        state_dict = torch.load(fam["seeds"][seed], map_location=DEVICE)
        model.load_state_dict(state_dict)
        model.eval()  
        _models[key] = model
    return _models[key]


def _thresholds_to_class_probs(p):
    """p: numpy array of shape (4,) — sigmoid outputs of the 4 CORAL
    thresholds for one image. Returns a (5,) array of class
    probabilities. Identical to the cp[] computation in notebook cell 8."""
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
    model_key: "<family>:<seed>" for a single seed (e.g. "e1b0:2024"), or
    "<family>:all" to average the CORAL threshold probabilities of every
    seed in that family (ensembling) before converting to class
    probabilities. Returns a dict ready for dr.html:
        predicted_class, predicted_grade, confidence,
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

    models = [_get_model(fam_key, s) for s in seeds]
    x = preprocess_bytes(image_bytes).to(DEVICE)

    per_seed_p = []
    with torch.no_grad():
        for model in models:
            logits = model(x)
            p = torch.sigmoid(logits).cpu().numpy()[0]  # shape (4,)
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