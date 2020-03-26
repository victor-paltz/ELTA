
import os

import pandas as pd

from preprocessing.preprocessing import preprocess, get_datasets_for_training
from code_to_submit import run_analysis


DATASET_FOLDER = "data"
CACHE_FOLDER = "cache"

os.makedirs(DATASET_FOLDER, exist_ok=True)
os.makedirs(CACHE_FOLDER, exist_ok=True)

print("\nTrainging pipeline\n")

##############################
# STEP 1: Cleaning the dataset
##############################

print("STEP 1: Cleaning the dataset...")

# take the preprocessed dateset in the cache folder
try:
    print("\t-> Loading cached dataset")
    X_train = pd.read_csv(
        f"{CACHE_FOLDER}/X_train_preprocessed.csv", index_col="Unnamed: 0")
    Y_train = pd.read_csv(
        f"{CACHE_FOLDER}/Y_train_preprocessed.csv", index_col="Unnamed: 0")

# If the dataset is not found
except:
    print("\t-> File not found, generating preprocessed datasets")
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

print("\t-> Done\n")


#############################
# STEP 2: sentences embedding
#############################

print("STEP 2: Preparing data for training...")

train_x, valid_x, train_y, valid_y = get_datasets_for_training(
    X_train['designation'], Y_train['prdtypecode'], "tfidf")

print("\t-> Done\n")

#############################
# STEP 3: Finding best hyper-parameters
#############################

print("STEP 3: Finding best hyper-parameters...")
print("\t-> Parameters hardcoded already")
print("\t-> Done\n")

#########################
# STEP 4: Printing scores
#########################

print("STEP 4: Printing scores...")

run_analysis(train_x, train_y, valid_x, valid_y)

print("\t-> Done\n")
