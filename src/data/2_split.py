import os
import random
import shutil

input_dirs = ["../../data/processed/images_resized", "../../data/processed/masks_resized"]
output_dirs = {
    "train": ["../../data/processed/train_images", "../../data/processed/train_masks"],
    "val": ["../../data/processed/val_images", "../../data/processed/val_masks"]
}

def split_data(input_dirs, output_dirs, split_ratio=0.9):
    image_files = os.listdir(input_dirs[0])
    mask_files = os.listdir(input_dirs[1])

    image_files = [f for f in image_files if f in mask_files]
    random.shuffle(image_files)

    train_size = int(len(image_files) * split_ratio)
    train_files = image_files[:train_size]
    val_files = image_files[train_size:]

    os.makedirs(output_dirs["train"][0], exist_ok=True)
    os.makedirs(output_dirs["train"][1], exist_ok=True)
    os.makedirs(output_dirs["val"][0], exist_ok=True)
    os.makedirs(output_dirs["val"][1], exist_ok=True)

    with open("../../data/processed/train.txt", "w") as train_txt, open("../../data/processed/val.txt", "w") as val_txt:
        for file in train_files:
            shutil.copy(os.path.join(input_dirs[0], file), os.path.join(output_dirs["train"][0], file))
            shutil.copy(os.path.join(input_dirs[1], file), os.path.join(output_dirs["train"][1], file))
            train_txt.write(file + "\n")
        
        for file in val_files:
            shutil.copy(os.path.join(input_dirs[0], file), os.path.join(output_dirs["val"][0], file))
            shutil.copy(os.path.join(input_dirs[1], file), os.path.join(output_dirs["val"][1], file))
            val_txt.write(file + "\n")

split_data(input_dirs, output_dirs)
print("Data splitting complete.")
