import cv2
import numpy as np
import tensorflow as tf
import os

MODEL_PATH = "models/unet_model.h5"
INPUT_DIR = "data/LEVIR-CD/test/images"
OUTPUT_DIR = "output/predictions"

os.makedirs(OUTPUT_DIR, exist_ok=True)

model = tf.keras.models.load_model(MODEL_PATH)


def preprocess(img):
    img = img / 255.0
    return np.expand_dims(img, axis=0)


for img_name in os.listdir(INPUT_DIR):
    img_path = os.path.join(INPUT_DIR, img_name)
    image = cv2.imread(img_path)

    prediction = model.predict(preprocess(image))[0]
    mask = (prediction > 0.5).astype(np.uint8) * 255

    cv2.imwrite(os.path.join(OUTPUT_DIR, img_name), mask)
