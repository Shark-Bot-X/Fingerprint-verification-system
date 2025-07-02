from flask import Flask, request, jsonify, send_file, send_from_directory
from flask_cors import CORS
import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from rembg import remove
from skimage.morphology import skeletonize
from fingerprint_enhancer import enhance_fingerprint

app = Flask(__name__)
CORS(app)
DATABASE = {
    "alice": "stored_fingerprints/alice_fingerprint.png",
    "sneh" : "stored_fingerprints/processed_1_mask.jpg"
}
#this is the database with name and location of respective fingerprint
# === Config ===
UPLOAD_FOLDER = 'uploads'
PROCESSED_FOLDER = 'processed'
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(PROCESSED_FOLDER, exist_ok=True)

# === Load Siamese Model ===
def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-9))

model = load_model(
    "siamese_fingerprint_model_v2.keras",
    custom_objects={'euclidean_distance': euclidean_distance}
)

def apply_rembg_white_background(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Could not read: {image_path}")
    output_rgba = remove(img)
    alpha = output_rgba[:, :, 3] / 255.0
    foreground = output_rgba[:, :, :3]
    white_bg = np.ones_like(foreground, dtype=np.uint8) * 255
    result = (foreground * alpha[..., None] + white_bg * (1 - alpha[..., None])).astype(np.uint8)
    return result

# === Fingerprint segmentation ===
def segment_finger(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    gray = clahe.apply(gray)
    enhanced = enhance_fingerprint(gray)
    if enhanced.dtype == bool:
        enhanced = (enhanced * 255).astype(np.uint8)

    _, mask = cv2.threshold(enhanced, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    return mask

# === Process single image with input and output ===
def preprocess(image_path):
    img_white_bg = apply_rembg_white_background(image_path)
    mask = segment_finger(img_white_bg)
    resized = cv2.resize(mask, (128, 128))
    normalized = resized.astype(np.float32) / 255.0
    return normalized[..., np.newaxis]

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

# === Routes ===

@app.route("/")
def serve_html():
    return send_file("verify.html")

@app.route("/processed/<filename>")
def serve_processed(filename):
    return send_from_directory(PROCESSED_FOLDER, filename)

@app.route("/verify", methods=["POST"])
def verify_fingerprint():
    name = request.form.get("name")
    if name not in DATABASE:
        return jsonify({"error": f"Name '{name}' not found in database."}), 404

    if 'file' not in request.files:
        return jsonify({"error": "No fingerprint file provided"}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({"error": "Empty filename."}), 400
    if not allowed_file(file.filename):
        return jsonify({"error": "Invalid file type."}), 400

    input_path = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(input_path)

    # Process input image and save mask for visualization
    processed_mask = segment_finger(apply_rembg_white_background(input_path))
    processed_mask_resized = cv2.resize(processed_mask, (128, 128))
    processed_filename = f"processed_{os.path.splitext(file.filename)[0]}_mask.jpg"
    processed_path = os.path.join(PROCESSED_FOLDER, processed_filename)
    cv2.imwrite(processed_path, processed_mask)

    # Preprocess both images for model
    img1 = preprocess(DATABASE[name])
    img2 = preprocess(input_path)
    img1 = np.expand_dims(img1, axis=0)
    img2 = np.expand_dims(img2, axis=0)

    # Predict similarity
    prediction = model.predict([img1, img2])[0][0]
    match = bool(prediction > 0.2)  # Lower distance = more similar
    print({
        "similarity": float(prediction),
        "match": match,
        "processed_mask": f"/processed/{os.path.basename(processed_path)}"
    })
    return jsonify({
        "similarity": float(prediction),
        "match": match,
        "processed_mask": f"/processed/{processed_filename}"
    })

if __name__ == "__main__":
    app.run(debug=True)
