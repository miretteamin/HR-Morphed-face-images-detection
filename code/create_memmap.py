import numpy as np
import os
import json
from PIL import Image
import tqdm


DIR_PATH = "/Users/igorgogarev/Documents/m2/idemia/data/Crop_96_2fp_eye"
CONFIG_PATH = "./configs/config.json"
MODE = "val_txt"  # "train_txt" or "val_txt"
MEMMAP_PATH = "/Users/igorgogarev/Documents/m2/idemia/data/val_memmap.dat"
LABELS_PATH = "/Users/igorgogarev/Documents/m2/idemia/data/val_labels.npy"

if __name__ == "__main__":
    with open(CONFIG_PATH, 'r') as file:
        config = json.load(file)

    image_paths = []
    labels = []

    for file_path in config[MODE]:
        with open(os.path.join(DIR_PATH, file_path), 'r') as file:
            for line in file:
                path, label = line.strip().split(' ')
                image_paths.append(path)
                labels.append(int(label))

    labels = np.array(labels)
    np.save(LABELS_PATH, labels)

    print(f"Total number of images found: {len(image_paths)}.")
    print(f"Negative class count: {np.sum(labels == 0)}.")
    print(f"Positive class count: {np.sum(labels == 1)}.")

    img = Image.open(os.path.join(DIR_PATH, image_paths[0])).convert('RGB')
    image_shape = np.array(img, dtype=np.uint8).shape

    memmap_file = np.memmap(MEMMAP_PATH, dtype=np.uint8, mode='w+', shape=(len(image_paths), *image_shape))

    for i, path in enumerate(tqdm.tqdm(image_paths)):
        img = Image.open(os.path.join(DIR_PATH, path)).convert('RGB')
        image = np.array(img, dtype=np.uint8)
        memmap_file[i] = image

    memmap_file.flush()

