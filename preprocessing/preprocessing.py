
from sklearn import model_selection, preprocessing
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from debuging.info_decorator import timer
from preprocessing.cleaning_functions import (raw_to_tokens,
                                              remove_unfrequent_words)


@timer
def preprocess(X, Y):

    # remove useless columns
    X = X[["designation", "productid"]]

    # tokenize and clean every line of the dataset
    X["designation"] = [raw_to_tokens(text)
                        for text in X["designation"]]

    # remove unfrequent words
    X = remove_unfrequent_words(X)

    # keep only not empty sentences
    mask = X["designation"].apply(len) > 0
    X = X[mask]
    Y = Y[mask]

    return X, Y


def get_datasets_for_training(X, Y, embedding_type="tfidf"):
    """
    Compute the tfidf or bow on the input dataset and transform the labels
    """

    assert embedding_type in ["tfidf", "bow"]

    vectorizer = TfidfVectorizer() if embedding_type == "tfidf" else CountVectorizer()

    X_vec = vectorizer.fit_transform(X)

    train_x, valid_x, train_y, valid_y = model_selection.train_test_split(
        X_vec, Y)
    encoder = preprocessing.LabelEncoder()
    train_y = encoder.fit_transform(train_y)
    valid_y = encoder.fit_transform(valid_y)

    return train_x, valid_x, train_y, valid_y
