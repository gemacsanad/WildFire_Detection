import os
import zipfile
import numpy as np
import cv2
import tensorflow as tf
from tensorflow import keras
from kaggle.api.kaggle_api_extended import KaggleApi


DATASET_NAME = "abdelghaniaaba/wildfire-prediction-dataset"  # Wildfire Prediction Dataset
DATA_PATH = "./data"

def download_dataset():
    expected_folders = ["train", "test", "valid"]
    
    # Check if all expected folders exist in DATA_PATH
    if all(os.path.exists(os.path.join(DATA_PATH, folder)) for folder in expected_folders):
        print("Dataset already exists. Skipping download.")
        return
    
    print("Dataset not found. Downloading now...")
    api = KaggleApi()
    api.authenticate()
    os.makedirs(DATA_PATH, exist_ok=True)
    api.dataset_download_files(DATASET_NAME, path=DATA_PATH, unzip=True)



def load_data(split):
    data_dir = os.path.join(DATA_PATH, split)  
    X, Y = [], []
    failed_paths = []

    # The split folder contains wildfire and nowildfire
    categories = {"nowildfire": 0, "wildfire": 1}

    for category, label in categories.items():
        folder_path = os.path.join(data_dir, category)

        if not os.path.exists(folder_path):
            print(f"Warning: Missing category folder {folder_path}, skipping...")
            continue

        print(f"Reading {split} {category} ")
        for img_name in os.listdir(folder_path):
            img_path = os.path.join(folder_path, img_name)

            try:
                image = cv2.imread(img_path, cv2.IMREAD_COLOR)

                if image is None:
                    print(f"Warning: cv2.imread failed for {img_path}, skipping...")
                    failed_paths.append(img_path)
                    continue

                X.append(img_path)
                Y.append(label)

            except Exception as e:
                print(f"Error processing image {img_path}: {e}, skipping...")
                failed_paths.append(img_path)
                continue

    return X, Y, failed_paths



def preprocess_data(X_paths, Y, image_size=(128, 128)):
    X = []
    
    for img_path in X_paths:
        try:
            image = cv2.imread(img_path, cv2.IMREAD_COLOR)
            if image is None:
                continue
            image = cv2.resize(image, image_size)
            image = image / 255.0  # Normalize pixel values
            X.append(image)
        except Exception as e:
            print(f"Error preprocessing {img_path}: {e}, skipping...")
            continue

    X = np.array(X, dtype=np.float32)
    Y = np.array(Y, dtype=np.int32)
    return X, Y

def augment_image(image):
    image = tf.image.random_flip_left_right(image)  # Random horizontal flip (p=0.5)
    image = tf.image.rot90(image, k=np.random.randint(0, 4))  # Random 90° rotations
    image = tf.image.random_crop(image, size=[100, 100, 3])  # Random resized crop
    image = tf.image.random_brightness(image, max_delta=0.2)  # Adjust brightness
    image = tf.image.random_contrast(image, lower=0.8, upper=1.2)  # Adjust contrast
    return image

download_dataset()

# Load and preprocess training data
X_train_paths, Y_train, failed_train_paths = load_data("train")
X_train, Y_train = preprocess_data(X_train_paths, Y_train)

# Load and preprocess testing data
X_test_paths, Y_test, failed_test_paths = load_data("test")
X_test, Y_test = preprocess_data(X_test_paths, Y_test)

# Load and preprocess validation data
X_valid_paths, Y_valid, failed_valid_paths = load_data("valid")
X_valid, Y_valid = preprocess_data(X_valid_paths, Y_valid)


X_train_augmented = np.array([augment_image(img).numpy() for img in X_train])
X_test_augmented = np.array([augment_image(img).numpy() for img in X_test])
X_valid_augmented = np.array([augment_image(img).numpy() for img in X_valid])

# Save processed data inside data directory
np.save(os.path.join(DATA_PATH, "X_train.npy"), X_train)
np.save(os.path.join(DATA_PATH, "Y_train.npy"), Y_train)
np.save(os.path.join(DATA_PATH, "X_test.npy"), X_test)
np.save(os.path.join(DATA_PATH, "Y_test.npy"), Y_test)
np.save(os.path.join(DATA_PATH, "X_valid.npy"), X_valid)
np.save(os.path.join(DATA_PATH, "Y_valid.npy"), Y_valid)

print(" Data preparation complete. X and Y matrices saved.")

print("\n--- Failed Image Paths ---")
all_failed_paths = failed_train_paths + failed_test_paths + failed_valid_paths
if all_failed_paths:
    for path in all_failed_paths:
        try:
            os.remove(path)
            print(f"Deleted: {path}")
        except OSError as e:
            print(f"Error deleting {path}: {e}")
else:
    print("No images failed to process.")
