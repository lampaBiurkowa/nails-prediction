import os
import numpy as np
import tensorflow as tf
from keras.utils import img_to_array
import matplotlib.pyplot as plt
from PIL import Image
import cv2

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def resize_image_aspect_ratio(image_path, target_size=(128, 128)):
    img = Image.open(image_path)
    img = img.convert("RGB")
    img.thumbnail(target_size, Image.LANCZOS)

    new_img = Image.new("RGB", target_size, (255, 255, 255))
    new_img.paste(img, ((target_size[0] - img.size[0]) // 2, (target_size[1] - img.size[1]) // 2))

    return new_img

def apply_clahe_to_color_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    
    enhanced_lab_image = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2RGB)
    return enhanced_image

def load_and_preprocess_image(image_path, target_size=(128, 128)):
    resized_image = resize_image_aspect_ratio(image_path, target_size)
    image = np.array(resized_image)
    
    image_with_clahe = apply_clahe_to_color_image(image)
    return np.expand_dims(image_with_clahe / 255.0, axis=0)

def predict(model, image_path):
    image = load_and_preprocess_image(image_path, target_size=(128, 128))
    prediction = model.predict(image)
    return prediction[0]

def display_prediction(prediction):
    print(np.min(prediction), np.max(prediction))
    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.title("Original Prediction")
    plt.imshow(prediction, cmap='hot')
    plt.axis('off')

    plt.subplot(1, 2, 2)
    plt.title("Thresholded Prediction (>= 0.185)")
    thresholded_prediction = np.where(prediction >= 0.185, prediction, 0)
    plt.imshow(thresholded_prediction, cmap='hot')
    plt.axis('off')

    plt.tight_layout()
    plt.show()

model = tf.keras.models.load_model('../../models/unet_model_128x128_2.keras', custom_objects={'combined_loss': combined_loss})
image_path = "../../data/test/4.jpg"
predicted_mask = predict(model, image_path)

display_prediction(predicted_mask)
