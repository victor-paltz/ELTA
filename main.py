
import os

import pandas as pd

from preprocessing import preprocess


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

"""from sklearn import model_selection, preprocessing, linear_model, naive_bayes, metrics
from sklearn import ensemble
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer

bow = CountVectorizer()
tfidf = TfidfVectorizer()

X_bow = bow.fit_transform(X_train['designation'])
X_tfidf = tfidf.fit_transform(X_train['designation'])


train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
    X_tfidf, Y_train['prdtypecode'])
encoder = preprocessing.LabelEncoder()
train_y = encoder.fit_transform(train_y)
valid_y = encoder.fit_transform(valid_y)
"""
