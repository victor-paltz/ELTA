
import os

import pandas as pd

from preprocessing import preprocess

DATASET_FOLDER = "data"
CACHE_FOLDER = "cache"

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

# take the preprocessed dateset in the cache folder
try:
    X_train = pd.read_csv(
        f"{CACHE_FOLDER}/X_train_preprocessed.csv", index_col='Unnamed: 0')
    Y_train = pd.read_csv(
        f"{CACHE_FOLDER}/Y_train_preprocessed.csv", index_col='Unnamed: 0')

# If the dataset is not found
except:

    # Load normal datasets
    X_train = pd.read_csv(
        f"{DATASET_FOLDER}/X_train_update.csv", index_col="Unnamed: 0")
    Y_train = pd.read_csv(
        f"{DATASET_FOLDER}/Y_train_CVw08PX.csv", index_col="Unnamed: 0")

    # preprocess datasets
    X_train, Y_train = preprocess(X_train, Y_train)

    # save preprocessed datasets
    X_train.to_csv(f"{CACHE_FOLDER}/X_train_preprocessed.csv")
    Y_train.to_csv(f"{CACHE_FOLDER}/Y_train_preprocessed.csv")

print("Done")
