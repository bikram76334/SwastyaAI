import base64

from flask import Flask, render_template, request

from inference import predict_dr, MODEL_CHOICES, DEFAULT_MODEL

# MODEL_CHOICES is grouped, so flatten it to get every valid option value
VALID_MODELS = {value for group in MODEL_CHOICES.values() for value in group}

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  # 10 MB only images

ALLOWED_TYPES = {"image/jpeg", "image/jpg", "image/png"}


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

# to add another module later like pneumonia: copy transforms.py/inference.py for that
# model's own preprocessing, architecture and predict function, then add its templates,
# form action, and copy the /dr route likewise for pneumonia ...

if __name__ == '__main__':
    app.run(debug=True)