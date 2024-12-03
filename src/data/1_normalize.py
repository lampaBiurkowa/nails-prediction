import os
from PIL import Image
import numpy as np


target_size = (128, 128)

def resize_image_with_aspect_ratio(image_path, target_size):
    try:
        img = Image.open(image_path)
        img = img.convert("RGB")
        width, height = img.size

        aspect_ratio = width / height
        if width > height:
            new_width = target_size[0]
            new_height = int(target_size[0] / aspect_ratio)
        else:
            new_height = target_size[1]
            new_width = int(target_size[1] * aspect_ratio)

        img_resized = img.resize((new_width, new_height), Image.ANTIALIAS)
        new_img = Image.new("RGB", target_size, (0, 0, 0))
        new_img.paste(img_resized, ((target_size[0] - new_width) // 2, (target_size[1] - new_height) // 2))

        return new_img
    except Exception as e:
        print(f"Error processing {image_path}: {e}")
        return None

def process(input_dir, output_dir):
    os.makedirs(output_dir, exist_ok=True)
    for image_name in os.listdir(input_dir):
        image_path = os.path.join(input_dir, image_name)
        
        if os.path.isfile(image_path) and image_name.endswith(".jpg"):
            resized_img = resize_image_with_aspect_ratio(image_path, target_size)

            if resized_img:
                output_path = os.path.join(output_dir, image_name)
                resized_img.save(output_path)
                print(f"Resized and saved: {image_name}")

process("../../data/processed/images", "../../data/processed/images_resized")
process("../../data/processed/masks", "../../data/processed/masks_resized")

print("Image resizing complete.")
