from flask import Flask, render_template, request
import pickle
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join(os.path.dirname(__file__), "house_model.pkl")

with open(MODEL_PATH, "rb") as f:
    model = pickle.load(f)

@app.route("/", methods=["GET", "POST"])
def home():
    prediction_text = None

    if request.method == "POST":
        sqft = float(request.form["sqft"])
        bedrooms = int(request.form["bedrooms"])
        bathrooms = int(request.form["bathrooms"])
        year_built = int(request.form["year"])

        features = np.array([[sqft, bedrooms, bathrooms, year_built]])
        prediction = round(model.predict(features)[0], 2)
        prediction = max(0, prediction)
        prediction_text = f"Estimated House Price: ₹ {prediction:,.2f}"

    return render_template("index.html", prediction_text=prediction_text)

if __name__ == "__main__":
    app.run()
