import numpy as np
import pandas as pd
from flask import Flask, render_template, request
import pickle
import os

app = Flask(__name__)

# Load model once at startup, using a path relative to this file
# (works locally AND on any deployment host, unlike a hardcoded C:\ or D:\ path)
MODEL_PATH = os.path.join(os.path.dirname(__file__), "HDI.pkl")
model = pickle.load(open(MODEL_PATH, "rb"))


@app.route("/")
def home():
    return render_template("home.html")


@app.route("/Prediction", methods=["POST", "GET"])
def prediction():
    return render_template("indexnew.html")


@app.route("/Home", methods=["POST", "GET"])
def my_home():
    return render_template("home.html")


@app.route("/predict", methods=["POST"])
def predict():
    input_features = [float(x) for x in request.form.values()]

    features_name = [
        "Country",
        "Life expectancy",
        "Mean years of schooling",
        "Gross national income (GNI) per capita",
        "Internet users",
    ]

    df = pd.DataFrame([input_features], columns=features_name)
    output = model.predict(df)
    y_pred = round(output[0][0], 2)

    if 0.3 <= y_pred <= 0.4:
        tier_class, tier_label = "low", "Low"
    elif 0.4 < y_pred <= 0.7:
        tier_class, tier_label = "medium", "Medium"
    elif 0.7 < y_pred <= 0.8:
        tier_class, tier_label = "high", "High"
    elif 0.8 < y_pred <= 0.94:
        tier_class, tier_label = "veryhigh", "Very High"
    else:
        tier_class, tier_label = None, None

    if tier_class is None:
        return render_template(
            "resultnew.html",
            in_range=False,
            value=y_pred,
        )

    marker_pct = max(0.0, min(1.0, y_pred)) * 100

    return render_template(
        "resultnew.html",
        in_range=True,
        value=y_pred,
        tier_class=tier_class,
        tier_label=tier_label,
        marker_pct=round(marker_pct, 1),
    )


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port, debug=False)
