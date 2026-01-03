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
    prediction = None

    if request.method == "POST":
        sqft = float(request.form["sqft"])
        bedroom = int(request.form["bedroom"])
        bathroom = int(request.form["bathroom"])
        year_built = int(request.form["year"])

        features = np.array([[sqft, bedroom, bathroom, year_built]])
        prediction = round(model.predict(features)[0], 2)

    return render_template("index.html", prediction=prediction)

if __name__ == "__main__":
    app.run()
