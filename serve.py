# serve.py
import os
from flask import Flask, request, jsonify, render_template
from PIL import Image
import io
from model_utils import load_model, load_labels, predict_image

app = Flask(__name__, static_folder="web/static", template_folder="web/templates")

# Load model on startup (CPU)
MODEL = load_model(device="cpu", quantized=True)
LABELS = load_labels()

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if 'image' not in request.files:
        return jsonify({"error": "no image provided"}), 400
    file = request.files['image']
    try:
        img = Image.open(file.stream).convert("RGB")
    except Exception as e:
        return jsonify({"error": "cannot open image", "details": str(e)}), 400

    probs, idxs = predict_image(MODEL, img, topk=3)
    results = []
    for p, i in zip(probs, idxs):
        label = LABELS.get(int(i), str(i))
        results.append({"label": label, "probability": float(p)})
    return jsonify({"predictions": results})

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
