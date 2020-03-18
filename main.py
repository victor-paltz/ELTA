
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


############################
# STEP 2: TFIDF on sentences
############################


"""from sklearn.feature_extraction.text import CountVectorizer
vectorizer = CountVectorizer()
X_BOW = vectorizer.fit_transform(X_train['designation'])
print(X_BOW.todense().shape)
print(vectorizer.get_feature_names()[:100])

from sklearn.feature_extraction.text import TfidfVectorizer
tfidf = TfidfVectorizer()
X_tfidf = tfidf.fit_transform(X_train['designation'])
print(X_tfidf.todense().shape)
print(tfidf.get_feature_names()[:100])

from sklearn import model_selection, preprocessing,linear_model, naive_bayes, metrics
from sklearn import ensemble"""
