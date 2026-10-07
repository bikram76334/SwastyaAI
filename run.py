import base64

from flask import Flask, render_template, request

from inference import predict_dr, MODEL_CHOICES

app=Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 10 * 1024 * 1024  #10 MB only images

ALLOWED_TYPES={"image/jpeg", "image/jpg", "image/png"}


@app.route('/')
def home():
    return render_template('home.html')


@app.route("/about")
def about():
    return render_template('about.html')


@app.route("/dr", methods=["GET", "POST"])
def dr():
    if request.method == "GET":
        return render_template("dr.html", model_choices=MODEL_CHOICES, selected_model="2024")

    selected_model = request.form.get("model", "2024")
    ctx = {"model_choices": MODEL_CHOICES, "selected_model": selected_model}

    if selected_model not in dict(MODEL_CHOICES):
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

       #Preview only  nothing is written to disk, matches the "no stored
#images" promise on the about page.
    result["image_data_uri"] = (
        "data:" + file.mimetype + ";base64," + base64.b64encode(image_bytes).decode("ascii")
    )

    return render_template("dr.html", result=result, **ctx)


@app.errorhandler(413)
def too_large(e):
    return render_template(
        "dr.html",
        error="That file is too large. Please upload an image under 10 MB.",
        model_choices=MODEL_CHOICES,
        selected_model="2024",
    ), 413
# to add antoher module later like pneuonia// we have to copy trnasforms.py/infernece.py for any model own preprocessing , archeteture and predict function 
# and simple add all templates,and form action, copy the /dr routes like wise to pnuemonia ...

if __name__ == '__main__':
    app.run(debug=True)