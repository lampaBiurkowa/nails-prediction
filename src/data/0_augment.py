import os
import shutil
import json
import cv2
import numpy as np
from PIL import Image, ImageDraw
import albumentations as album

# Paths
old_image_dir = "../../data/external/images"
old_masks_dir = "../../data/external/labels"
new_image_dirs = ["../../data/external/test", "../../data/external/train", "../../data/external/valid"]
target_image_dir = "../../data/processed/images"
target_masks_dir = "../../data/processed/masks"

# Prepare target directories
if os.path.exists(target_image_dir):
    shutil.rmtree(target_image_dir)
os.makedirs(target_image_dir, exist_ok=True)

if os.path.exists(target_masks_dir):
    shutil.rmtree(target_masks_dir)
os.makedirs(target_masks_dir, exist_ok=True)

def get_training_augmentation():
    train_transform = [
        album.RandomCrop(height=128, width=128, always_apply=True),
        album.OneOf(
            [
                album.HorizontalFlip(p=1),
                album.VerticalFlip(p=1),
                album.RandomRotate90(p=1),
            ],
            p=0.75,
        ),
    ]
    return album.Compose(train_transform)

def apply_clahe(image):
    clahe = album.CLAHE(clip_limit=3.0, tile_grid_size=(8, 8), always_apply=True)
    return clahe(image=image)['image']

def create_mask_from_annotations(image_size, polygons):
    mask = Image.new("L", image_size, 0)
    draw = ImageDraw.Draw(mask)
    for polygon in polygons:
        draw.polygon(polygon, outline=1, fill=1)
    return np.array(mask) * 255

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

            image_clahe = apply_clahe(image)

            original_image_name = f"{os.path.splitext(image_name)[0]}_original{os.path.splitext(image_name)[1]}"
            cv2.imwrite(os.path.join(target_image_dir, original_image_name), image_clahe)

            image_id = next((img_id for img_id, img in image_data.items() if img['file_name'] == image_name), None)
            if image_id is not None:
                annotations_for_image = [ann for ann in annotation_data if ann['image_id'] == image_id]
                height, width = image.shape[:2]
                polygons = [ann['segmentation'][0] for ann in annotations_for_image if 'segmentation' in ann]
                polygons = [[tuple(coord) for coord in zip(polygon[::2], polygon[1::2])] for polygon in polygons]
                mask = create_mask_from_annotations((width, height), polygons)

                original_mask_name = f"{os.path.splitext(image_name)[0]}_original.jpg"
                cv2.imwrite(os.path.join(target_masks_dir, original_mask_name), mask)

                augmented = get_training_augmentation()(image=image_clahe, mask=mask)
                image_aug, mask_aug = augmented['image'], augmented['mask']

                augmented_image_name = f"{os.path.splitext(image_name)[0]}_augmented{os.path.splitext(image_name)[1]}"
                augmented_mask_name = f"{os.path.splitext(image_name)[0]}_augmented.jpg"
                cv2.imwrite(os.path.join(target_image_dir, augmented_image_name), image_aug)
                cv2.imwrite(os.path.join(target_masks_dir, augmented_mask_name), mask_aug)

print("Processing and augmentation completed.")
