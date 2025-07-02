import os
import cv2
import random
import numpy as np
import tensorflow as tf
from collections import Counter
from sklearn.model_selection import train_test_split
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Conv2D, MaxPooling2D, Flatten, Dense, Lambda, LayerNormalization
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.utils import Sequence
import matplotlib.pyplot as plt

# ------------ CONFIG ----------------
IMAGE_SIZE = (128, 128)
DATASET_PATH = r"C:\Users\SNEH\Desktop\Fingerprint-sensor\Finger"
EPOCHS = 8
BATCH_SIZE = 8
MODEL_SAVE_PATH = "siamese_fingerprint_model_v2.keras"
# ------------------------------------

def preprocess(image_path):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise FileNotFoundError(f"Image not found: {image_path}")
    img = cv2.resize(img, IMAGE_SIZE)
    img = cv2.equalizeHist(img)
    img = img.astype("float32") / 255.0
    return img[..., np.newaxis]

def augment(img):
    # Optional: You can extend this with flips, rotations, etc.
    return img

class SiameseDataGen(Sequence):
    def __init__(self, pairs, labels, batch_size=8, augment_enabled=True):
        self.pairs = pairs
        self.labels = labels
        self.batch_size = batch_size
        self.augment_enabled = augment_enabled

    def __len__(self):
        return int(np.ceil(len(self.pairs) / self.batch_size))

    def __getitem__(self, idx):
        batch_pairs = self.pairs[idx*self.batch_size:(idx+1)*self.batch_size]
        batch_labels = self.labels[idx*self.batch_size:(idx+1)*self.batch_size]
        x1, x2 = [], []

        for img1_path, img2_path in batch_pairs:
            img1 = preprocess(img1_path)
            img2 = preprocess(img2_path)
            if self.augment_enabled:
                img1 = augment(img1)
                img2 = augment(img2)
            x1.append(img1)
            x2.append(img2)

        return (np.array(x1), np.array(x2)), np.array(batch_labels)

def euclidean_distance(vects):
    x, y = vects
    return tf.sqrt(tf.maximum(tf.reduce_sum(tf.square(x - y), axis=1, keepdims=True), 1e-9))

def build_base_cnn():
    input_img = Input((*IMAGE_SIZE, 1))
    x = Conv2D(16, (3, 3), activation='relu', padding='same')(input_img)
    x = MaxPooling2D()(x)
    x = Conv2D(32, (3, 3), activation='relu', padding='same')(x)
    x = MaxPooling2D()(x)
    x = Flatten()(x)
    x = Dense(64, activation='relu')(x)
    x = LayerNormalization()(x)
    return Model(input_img, x)

def build_siamese_network():
    input_a = Input((*IMAGE_SIZE, 1))
    input_b = Input((*IMAGE_SIZE, 1))

    base_model = build_base_cnn()
    feature_a = base_model(input_a)
    feature_b = base_model(input_b)

    distance = Lambda(euclidean_distance)([feature_a, feature_b])
    output = Dense(1, activation='sigmoid')(distance)

    return Model(inputs=[input_a, input_b], outputs=output)

def create_pairs_balanced(dataset_path):
    persons = [d for d in os.listdir(dataset_path) if os.path.isdir(os.path.join(dataset_path, d))]
    persons.sort()

    image_dict = {}
    for person in persons:
        person_path = os.path.join(dataset_path, person)
        images = [os.path.join(person_path, img)
                  for img in os.listdir(person_path)
                  if img.lower().endswith(('.png', '.jpg', '.jpeg'))]
        if len(images) >= 2:
            image_dict[person] = images

    pairs = []
    labels = []

    # Positive pairs
    for person, imgs in image_dict.items():
        for i in range(len(imgs)):
            for j in range(i + 1, len(imgs)):
                pairs.append((imgs[i], imgs[j]))
                labels.append(1)

    # Negative pairs
    person_list = list(image_dict.keys())
    for _ in range(len(pairs)):
        p1, p2 = random.sample(person_list, 2)
        img1 = random.choice(image_dict[p1])
        img2 = random.choice(image_dict[p2])
        pairs.append((img1, img2))
        labels.append(0)

    return pairs, labels

def train_model():
    print("📂 Creating balanced image pairs...")
    pairs, labels = create_pairs_balanced(DATASET_PATH)
    print("📊 Label distribution:", Counter(labels))

    train_pairs, val_pairs, train_labels, val_labels = train_test_split(
        pairs, labels, test_size=0.3, random_state=42)

    train_gen = SiameseDataGen(train_pairs, train_labels, batch_size=BATCH_SIZE)
    val_gen = SiameseDataGen(val_pairs, val_labels, batch_size=BATCH_SIZE, augment_enabled=False)

    model = build_siamese_network()
    model.compile(optimizer='adam', loss='binary_crossentropy', metrics=['accuracy'])

    print("🧠 Training model...")
    model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=[EarlyStopping(patience=3, restore_best_weights=True)]
    )

    model.save(MODEL_SAVE_PATH)
    print(f"✅ Model saved to: {MODEL_SAVE_PATH}")
    return model, val_pairs, val_labels

if __name__ == "__main__":
    model, val_pairs, val_labels = train_model()
