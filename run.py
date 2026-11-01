import base64

from flask import Flask, render_template, request, jsonify

from inference import predict_dr, MODEL_CHOICES, DEFAULT_MODEL
import requests,os ,threading,json
from concurrent.futures import ThreadPoolExecutor


# MODEL_CHOICES is grouped, so flatten it to get every valid option value
VALID_MODELS = {value for group in MODEL_CHOICES.values() for value in group}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB only images

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}

MYMEMORY_URL = "https://api.mymemory.translated.net/get"
CONTACT_EMAIL = "deepapaneru220@gmail.com"  # raises the daily limit to ~50k chars
CACHE_FILE = "translation_cache.json"

_lock = threading.Lock()

# load saved translations so a restart doesn't waste quota
try:
    with open(CACHE_FILE, "r", encoding="utf-8") as f:
        _cache = json.load(f)
except (FileNotFoundError, json.JSONDecodeError):
    _cache = {}


def save_cache():
    with _lock:
        with open(CACHE_FILE, "w", encoding="utf-8") as f:
            json.dump(_cache, f, ensure_ascii=False, indent=2)


def translate_one(text, target):
    """Translate one string with MyMemory. Falls back to the original on any problem."""
    key = f"{target}|{text}"
    if key in _cache:
        return _cache[key]

    # MyMemory allows max 500 chars per request
    if len(text) > 500:
        return text

    try:
        resp = requests.get(
            MYMEMORY_URL,
            params={"q": text, "langpair": f"en|{target}", "de": CONTACT_EMAIL},
            timeout=10,
        )
        data = resp.json()

        # responseStatus is 200 on success; quota errors come back as other codes
        if str(data.get("responseStatus")) != "200":
            return text

        translated = data["responseData"]["translatedText"]

        # quota warnings sometimes arrive as the "translation" text
        if not translated or "MYMEMORY WARNING" in translated.upper():
            return text

        _cache[key] = translated
        return translated
    except (requests.RequestException, ValueError, KeyError):
        return text


@app.route("/translate", methods=["POST"])
def translate():
    data = request.get_json(silent=True) or {}
    texts = data.get("texts", [])
    target = data.get("target", "ne")

    if not isinstance(texts, list) or not texts or len(texts) > 128:
        return jsonify(error="texts must be a list of 1-128 strings"), 400

    unique = list(dict.fromkeys(texts))

    # a few parallel requests so the page doesn't take forever
    with ThreadPoolExecutor(max_workers=5) as pool:
        results = list(pool.map(lambda t: translate_one(t, target), unique))

    lookup = dict(zip(unique, results))
    save_cache()

    return jsonify(translations=[lookup[t] for t in texts])



#home 
@app.route('/')
def home():
    return render_template('home.html')

@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/dr", methods=["GET", "POST"])
def dr():
    if request.method == "GET":
        return render_template("dr.html", model_choices=MODEL_CHOICES, selected_model=DEFAULT_MODEL)

    selected_model = request.form.get("model", DEFAULT_MODEL)
    ctx = {"model_choices": MODEL_CHOICES, "selected_model": selected_model}

    if selected_model not in VALID_MODELS:
        return render_template("dr.html", error="Please select a model.", **ctx)

    file = request.files.get("image")
    if file is None or file.filename == "":
        return render_template("dr.html", error="Please choose an image.", **ctx)
    if file.mimetype not in ALLOWED_TYPES:
        return render_template("dr.html", error="Please upload a JPG or PNG image.", **ctx)

    image_bytes = file.read()

    try:
        result = predict_dr(image_bytes, selected_model)
    except Exception:
        app.logger.exception("DR inference failed")
        return render_template(
            "dr.html",
            error="Could not process this image. Please try a different file.",
            **ctx,
        )

    
    # images display of run inference
    result["image_data_uri"] = (
        "data:" + file.mimetype + ";base64," + base64.b64encode(image_bytes).decode("ascii")
    )

    return render_template("dr.html", result=result, **ctx)


# for DR grade library
@app.route('/grade0')
def grade0():
    return render_template('grade0.html')

@app.route('/grade1')



def grade1():
    return render_template('grade1.html')

@app.route('/grade2')
def grade2():
    return render_template('grade2.html')

@app.route('/grade3')
def grade3():
    return render_template('grade3.html')

@app.route('/grade4')
def grade4():
    return render_template('grade4.html')


@app.errorhandler(413)
def too_large(e):
    return render_template(
        "dr.html",
        error="That file is too large. Please upload an image under 10 MB.",
        model_choices=MODEL_CHOICES,
        selected_model=DEFAULT_MODEL,
    ), 413


#Ear Disease

@app.route("/ear_disease")
def ear_disease():
    return render_template("ear.html")

# to add another module later like pneumonia: copy transforms.py/inference.py for that
# model's own preprocessing, architecture and predict function, then add its templates,
# form action, and copy the /dr route likewise for pneumonia ...

if __name__ == '__main__':
    app.run(debug=True)