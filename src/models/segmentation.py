import os
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.utils import load_img, img_to_array
import matplotlib.pyplot as plt

def dice_loss(y_true, y_pred):
    smooth = 1e-6
    intersection = tf.reduce_sum(y_true * y_pred)
    union = tf.reduce_sum(y_true) + tf.reduce_sum(y_pred)
    return 1 - (2. * intersection + smooth) / (union + smooth)
def combined_loss(y_true, y_pred):
    bce = tf.keras.losses.binary_crossentropy(y_true, y_pred)
    dice = dice_loss(y_true, y_pred)
    return bce + dice

def unet_model(input_size=(128, 128, 3), num_classes=1):
    inputs = layers.Input(input_size)

    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(inputs)
    c1 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c1)
    p1 = layers.MaxPooling2D((2, 2))(c1)

    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(p1)
    c2 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c2)
    p2 = layers.MaxPooling2D((2, 2))(c2)
    
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(p2)
    c3 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c3)
    p3 = layers.MaxPooling2D((2, 2))(c3)

    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(p3)
    c4 = layers.Conv2D(512, (3, 3), activation='relu', padding='same')(c4)

    u7 = layers.UpSampling2D((2, 2))(c4)
    u7 = layers.Conv2D(256, (2, 2), activation='relu', padding='same')(u7)
    u7 = layers.concatenate([u7, c3])
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(u7)
    c7 = layers.Conv2D(256, (3, 3), activation='relu', padding='same')(c7)

    u8 = layers.UpSampling2D((2, 2))(c7)
    u8 = layers.Conv2D(128, (2, 2), activation='relu', padding='same')(u8)
    u8 = layers.concatenate([u8, c2])
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(u8)
    c8 = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(c8)

    u9 = layers.UpSampling2D((2, 2))(c8)
    u9 = layers.Conv2D(64, (2, 2), activation='relu', padding='same')(u9)
    u9 = layers.concatenate([u9, c1])
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(u9)
    c9 = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(c9)

    outputs = layers.Conv2D(num_classes, (1, 1), activation='sigmoid')(c9)
    model = models.Model(inputs=[inputs], outputs=[outputs])

    return model

input_size = (128, 128, 3)
num_classes = 1
model = unet_model(input_size=input_size, num_classes=num_classes)

model.compile(optimizer='adam', loss=combined_loss, metrics=['accuracy'])
model.summary()

def load_image_mask(image_path, mask_path, target_size=(128, 128)):
    image = load_img(image_path, target_size=target_size)
    mask = load_img(mask_path, target_size=target_size, color_mode='grayscale')
    image = img_to_array(image) / 255.0
    mask = img_to_array(mask) / 255.0
    return image, mask

def get_image_mask_paths(images_dir, masks_dir):
    image_paths = [os.path.join(images_dir, fname) for fname in os.listdir(images_dir) if fname.endswith('.jpg')]
    mask_paths = [os.path.join(masks_dir, fname) for fname in os.listdir(masks_dir) if fname.endswith('.jpg')]
    return image_paths, mask_paths

train_images_dir = "../../data/processed/train_images"
train_masks_dir = "../../data/processed/train_masks"
val_images_dir = "../../data/processed/val_images"
val_masks_dir = "../../data/processed/val_masks"

train_image_paths, train_mask_paths = get_image_mask_paths(train_images_dir, train_masks_dir)
val_image_paths, val_mask_paths = get_image_mask_paths(val_images_dir, val_masks_dir)

train_images = np.array([load_image_mask(img, mask)[0] for img, mask in zip(train_image_paths, train_mask_paths)])
train_masks = np.array([load_image_mask(img, mask)[1] for img, mask in zip(train_image_paths, train_mask_paths)])
val_images = np.array([load_image_mask(img, mask)[0] for img, mask in zip(val_image_paths, val_mask_paths)])
val_masks = np.array([load_image_mask(img, mask)[1] for img, mask in zip(val_image_paths, val_mask_paths)])

history = model.fit(train_images, train_masks, 
                    validation_data=(val_images, val_masks), 
                    batch_size=32, epochs=10)

model.save('../../models/unet_model_128x128_2.keras')

# # Display Training Results
# plt.plot(history.history['loss'], label='Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.title('Loss Over Epochs')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()
