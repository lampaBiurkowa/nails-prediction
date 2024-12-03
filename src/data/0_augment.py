import os
import shutil
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw

old_image_dir = "../../data/external/images"
old_masks_dir = "../../data/external/labels"
new_image_dirs = ["../../data/external/test", "../../data/external/train", "../../data/external/valid"]
target_image_dir = "../../data/processed/images"
target_masks_dir = "../../data/processed/masks"

if os.path.exists(target_image_dir):
    shutil.rmtree(target_image_dir)
os.makedirs(target_image_dir, exist_ok=True)

if os.path.exists(target_masks_dir):
    shutil.rmtree(target_masks_dir)
os.makedirs(target_masks_dir, exist_ok=True)

def apply_clahe_to_color_image(image):
    lab_image = cv2.cvtColor(image, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab_image)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    enhanced_lab_image = cv2.merge((cl, a, b))
    enhanced_image = cv2.cvtColor(enhanced_lab_image, cv2.COLOR_LAB2BGR)
    return enhanced_image

def create_mask_from_annotations(image_size, polygons):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask) * 255

old_image_filenames = os.listdir(old_image_dir)
old_mask_filenames = os.listdir(old_masks_dir)

for image_name, mask_name in zip(old_image_filenames, old_mask_filenames):
    image_path = os.path.join(old_image_dir, image_name)
    mask_path = os.path.join(old_masks_dir, mask_name)

    image = cv2.imread(image_path)
    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    if image is None or mask is None:
        continue

    image_with_clahe = apply_clahe_to_color_image(image)

    new_image_name = f"{os.path.splitext(image_name)[0]}{os.path.splitext(image_name)[1]}"
    new_mask_name = f"{os.path.splitext(mask_name)[0]}{os.path.splitext(mask_name)[1]}"
    cv2.imwrite(os.path.join(target_image_dir, new_image_name), image_with_clahe)
    cv2.imwrite(os.path.join(target_masks_dir, new_mask_name), mask)

for dir_path in new_image_dirs:
    annotation_file = os.path.join(dir_path, "_annotations.coco.json")
    
    with open(annotation_file, 'r') as f:
        annotations = json.load(f)

    image_data = {img['id']: img for img in annotations['images']}
    annotation_data = annotations['annotations']

    for file in os.listdir(dir_path):
        if file.lower().endswith(('.png', '.jpg', '.jpeg')) and file != "_annotations.coco.json":
            image_path = os.path.join(dir_path, file)
            image_name = os.path.basename(image_path)
            image = cv2.imread(image_path)

            if image is None:
                continue

            image_with_clahe = apply_clahe_to_color_image(image)
            image_id = next((img_id for img_id, img in image_data.items() if img['file_name'] == image_name), None)
            if image_id is not None:
                annotations_for_image = [ann for ann in annotation_data if ann['image_id'] == image_id]
                height, width = image.shape[:2]
                polygons = [ann['segmentation'][0] for ann in annotations_for_image if 'segmentation' in ann]
                polygons = [[tuple(coord) for coord in zip(polygon[::2], polygon[1::2])] for polygon in polygons]
                mask = create_mask_from_annotations((width, height), polygons)
                new_image_name = f"{os.path.splitext(image_name)[0]}{os.path.splitext(image_name)[1]}"
                new_mask_name = f"{os.path.splitext(image_name)[0]}.jpg"
                cv2.imwrite(os.path.join(target_image_dir, new_image_name), image_with_clahe)
                cv2.imwrite(os.path.join(target_masks_dir, new_mask_name), mask)

print("Integration of datasets completed.")
