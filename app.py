import os
from pathlib import Path

import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request
from tensorflow.keras.preprocessing.image import img_to_array, load_img
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
UPLOAD_FOLDER = BASE_DIR / "static" / "uploads"
ALLOWED_EXTENSIONS = {"png", "jpg", "jpeg"}

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = 8 * 1024 * 1024  # 8 MB upload limit

# Load the saved model
model = tf.keras.models.load_model(BASE_DIR / "cats_vs_dogs_model.h5", compile=False)


# Function to preprocess the image
def preprocess_image(image_path, img_size=(64, 64)):
    img = load_img(image_path, target_size=img_size)
    img_array = img_to_array(img) / 255.0          # normalise to [0, 1] as in training
    return np.expand_dims(img_array, axis=0)       # add batch dimension


# Function to make predictions
def make_prediction(image_path):
    prob_dog = float(model.predict(preprocess_image(image_path), verbose=0)[0][0])
    label = "Dog" if prob_dog > 0.5 else "Cat"
    confidence = prob_dog if label == "Dog" else 1 - prob_dog
    return label, confidence


# Function to check allowed file types
def allowed_file(filename):
    return "." in filename and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS


# Homepage route
@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        file = request.files.get("file")
        if file is None or file.filename == "" or not allowed_file(file.filename):
            return render_template("index.html", error="Please upload a PNG or JPG image.")
        filename = secure_filename(file.filename)
        os.makedirs(UPLOAD_FOLDER, exist_ok=True)
        file.save(UPLOAD_FOLDER / filename)
        result, confidence = make_prediction(UPLOAD_FOLDER / filename)
        return render_template("index.html", result=result, confidence=confidence,
                               image_url=f"uploads/{filename}")
    return render_template("index.html")


if __name__ == "__main__":
    app.run(debug=True)
