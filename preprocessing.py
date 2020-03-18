
from cleaning_functions import raw_to_tokens, remove_unfrequent_words
from debuging.info_decorator import timer


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
