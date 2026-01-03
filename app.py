from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load trained model
model = pickle.load(open("house_model.pkl", "rb"))

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    sqft = float(request.form['sqft'])
    bedrooms = int(request.form['bedrooms'])
    bathrooms = int(request.form['bathrooms'])
    year_built = int(request.form['year'])

    features = np.array([[sqft, bedrooms, bathrooms, year_built]])
    prediction = model.predict(features)[0]
    prediction = max(0, prediction)

    return render_template(
        'index.html',
        prediction_text=f"Estimated House Price: ₹ {prediction:,.2f}"
    )

if __name__ == "__main__":
    app.run(debug=True)
