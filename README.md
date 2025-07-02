# 🧬 Fingerprint Verification System using Siamese Neural Network

A production-ready biometric authentication system built with Flask and TensorFlow. It uses a Siamese neural network to compute fingerprint similarity, combining advanced preprocessing (Rembg, OpenCV, CLAHE, Otsu thresholding) with a web-based interface.

---

## 🚀 Overview

This project enables secure fingerprint verification by comparing uploaded images against stored templates using deep learning. The core architecture leverages a Siamese network trained to measure similarity via Euclidean distance.

- 📦 Backend: Flask + TensorFlow
- 🧠 Model: Siamese Network (contrastive loss)
- 🎛️ Preprocessing: Rembg, OpenCV, fingerprint-enhancer
- 🌐 Interface: HTML + JS + REST API
- 📊 Output: Similarity score + binary match decision

---

## 🗂️ Project Structure

Fingerprint-sensor/
├── server.py # Main Flask application
├── verify.html # UI interface (HTML/JS)
├── train_siamese.py # Model training script (optional)
├── siamese_fingerprint_model_v2.keras # Trained model
├── stored_fingerprints/ # Stored template fingerprints
├── uploads/ # Temporary uploaded images
├── processed/ # Processed fingerprint masks
└── README.md

---

## 🧪 Model Architecture

- Siamese CNN with shared weights
- Input shape: `(128, 128, 1)`
- Distance metric: Euclidean
- Threshold-based classification (e.g., `distance < 0.3 → match`)
- Custom loss: `ContrastiveLoss(margin=1.0)`

---

## 🖥️ Installation

### 1. Clone the Repository

bash
git clone https://github.com/Shark-Bot-X/Fingerprint-verification-system
cd fingerprint-verification-app
2. Install Dependencies
pip install -r requirements.txt
<details> <summary>Dependencies include:</summary>
flask

flask-cors

tensorflow

opencv-python

rembg

numpy

scikit-image

fingerprint-enhancer

</details>
▶️ Running the Application
python server.py
Visit http://127.0.0.1:5000 in your browser.

⚙️ API Endpoint
POST /verify
Payload: multipart/form-data

Field	Type	Description
name	string	Identifier of the stored template
file	file	Fingerprint image (.jpg/.png)

Response:
{
  "similarity": 0.276,
  "match": true,
  "processed_mask": "/processed/processed_input_mask.jpg"
}
🔍 Fingerprint Preprocessing Pipeline
Background Removal → rembg

CLAHE Normalization → Adaptive histogram equalization

Enhancement → fingerprint-enhancer

Binarization → Otsu thresholding

Resize + Normalize → Final (128, 128, 1) tensor

📈 Model Training (Optional)
To train your own model on custom fingerprint pairs, use:

python train_siamese.py
Customize:

Dataset structure

Augmentation pipeline

Threshold tuning for optimal accuracy

📸 Sample UI
![image](https://github.com/user-attachments/assets/12cca5be-9941-4ba9-9800-7b290370dfe6)

📌 Use Cases
Identity verification systems

Access control/authentication terminals

Biometric research experiments

Forensics & digital evidence validation

📄 License
This project is licensed under the MIT License.
Feel free to use, modify, and distribute with attribution.

