"""
Python script to submit as a part of the project of ELTP 2020 course.

Group Members:
    (1) Chu Haotian
    (2) Debdeep Roy
    (3) Paltz Victor

"""
import time

import numpy as np
import xgboost
from sklearn import ensemble, linear_model, metrics, naive_bayes, tree
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score


def model_random_forest(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = RandomForestClassifier(min_samples_split=2,
                                 n_estimators=500,
                                 max_depth=None)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return acc, f1


def model_LogisticRegression(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = linear_model.LogisticRegression(C=2, penalty="l2")
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return acc, f1


def model_MultinomialNB(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = naive_bayes.MultinomialNB(alpha=0.1)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return acc, f1


def model_GradientBoosting(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = ensemble.GradientBoostingClassifier(
        n_estimators=1000, learning_rate=0.2, max_depth=3)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return acc, f1


def model_AdaBoost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = ensemble.AdaBoostClassifier(base_estimator=tree.DecisionTreeClassifier(max_depth=2),
                                      learning_rate=.1,
                                      n_estimators=2000)
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return acc, f1


def model_XGBoost(X_train, y_train, X_test, y_test):
    """
    @param: X_train - a numpy matrix containing features for training data (e.g. TF-IDF matrix)
    @param: y_train - a numpy array containing labels for each training sample
    @param: X_test - a numpy matrix containing features for test data (e.g. TF-IDF matrix)
    @param: y_test - a numpy array containing labels for each test sample
    """
    clf = xgboost.XGBClassifier()
    clf.fit(X_train, y_train)

    y_predicted = clf.predict(X_test)
    acc = accuracy_score(y_test, y_predicted)
    f1 = f1_score(y_test, y_predicted, average="weighted")

    return acc, f1


def run_analysis(X_train, y_train, X_test, y_test):
    """
    Compute the scores of all the models
    """

    model_names = ["Random Forest",
                   "Logistic Regression",
                   "Multinomial Naive Bayes",
                   "Gradient Boosting",
                   "AdaBoost",
                   "XGBoost"]

    model_functions_list = [model_random_forest,
                            model_LogisticRegression,
                            model_MultinomialNB,
                            model_GradientBoosting,
                            model_AdaBoost,
                            model_XGBoost]

    for run_model, model_name in zip(model_functions_list, model_names):
        print("Evaluating model", model_name, ":")
        begining = time.perf_counter()
        accuracy, weighted_f1 = run_model(X_train, y_train, X_test, y_test)
        print(f" -> Weighted F1 score: {weighted_f1:.3f}")
        print(f" -> Accuracy score: {accuracy:.3f}")
        print(f" -> training time: {time.perf_counter()-begining:.1f} seconds")


if __name__ == "__main__":

    # /!\ /!\ /!\
    # Load here X_train, y_train, X_test, y_test
    # /!\ /!\ /!\

    X_train = np.array([[1], [2]])
    y_train = np.array([0, 1])
    X_test = np.array([[1]])
    y_test = np.array([0])

    # In order to use the data we preprocessed, launch
    # the main.py file on our repository instead of this file.
    # It will launch the whole pipeline (5 minutes of preprocessing).

    run_analysis(X_train, y_train, X_test, y_test)
