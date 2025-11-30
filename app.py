from flask import Flask, request, jsonify
import numpy as np
import joblib
from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Load models
models = joblib.load("career_models.pkl")

label_cols = [
    "offensive", "blue_team", "malware", "forensics",
    "network", "cloud", "appsec", "threatintel", "grc"
]

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json

    # Ambil 20 jawaban
    answers = np.array([data[f"q{i}"] for i in range(1, 21)]).reshape(1, -1)

    predictions = {}

    # Predict tiap career
    for label in label_cols:
        model = models[label]
        prob = model.predict_proba(answers)[0][1]
        predictions[label] = float(prob)

    # Ambil top-3
    top3 = sorted(predictions.items(), key=lambda x: x[1], reverse=True)[:3]

    return jsonify({
        "probabilities": predictions,
        "top_3_career_recommendation": top3
    })

if __name__ == "__main__":
    app.run(port=5000, debug=True)
